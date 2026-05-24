import argparse
import os
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# VTK reader (ASCII STRUCTURED_POINTS, SCALARS blocks)
# Reads only requested scalar fields
# ============================================================
def _read_n_floats_stream(f, n, store=True):
    out = np.empty(n, dtype=np.float64) if store else None
    filled = 0
    while filled < n:
        line = f.readline()
        if not line:
            raise EOFError("Unexpected EOF while reading scalar data.")
        vals = np.fromstring(line, sep=" ", dtype=np.float64)
        if vals.size == 0:
            continue
        take = min(vals.size, n - filled)
        if store:
            out[filled:filled + take] = vals[:take]
        filled += take
    return out


def read_vtk_structured_points_ascii(path, want_names):
    want = set(want_names)
    with open(path, "r") as f:
        nx = ny = nz = None
        npts = None

        # Header scan
        for line in f:
            s = line.strip()
            if s.startswith("DIMENSIONS"):
                _, a, b, c = s.split()[:4]
                nx, ny, nz = int(a), int(b), int(c)
            elif s.startswith("POINT_DATA"):
                npts = int(s.split()[1])
                break

        if nx is None or npts is None:
            raise RuntimeError("Could not parse DIMENSIONS / POINT_DATA.")
        if npts != nx * ny * nz:
            raise RuntimeError("POINT_DATA != nx*ny*nz. Unexpected VTK layout.")

        fields = {}
        while True:
            line = f.readline()
            if not line:
                break
            s = line.strip()
            if not s:
                continue
            if s.startswith("SCALARS"):
                parts = s.split()
                name = parts[1]
                _ = f.readline()  # LOOKUP_TABLE default

                store = (name in want)
                vals = _read_n_floats_stream(f, npts, store=store)
                if store:
                    fields[name] = vals.reshape((nx, ny, nz), order="F")

        missing = [k for k in want_names if k not in fields]
        if missing:
            raise RuntimeError(f"Missing fields in VTK: {missing}")

        return (nx, ny, nz), fields


# ============================================================
# Axis parsing for RD/TD/ND
# ============================================================
def parse_axis(tok: str) -> np.ndarray:
    tok = tok.strip().lower()
    sgn = 1.0
    if tok.startswith("-"):
        sgn = -1.0
        tok = tok[1:]
    if tok not in ("x", "y", "z"):
        raise ValueError("Axis must be one of x,y,z,-x,-y,-z")
    v = {"x": np.array([1.0, 0.0, 0.0]),
         "y": np.array([0.0, 1.0, 0.0]),
         "z": np.array([0.0, 0.0, 1.0])}[tok]
    return sgn * v


# ============================================================
# Build R_sc from selected voxels
# ============================================================
def build_R_from_fields(fields, idx_flat, shape_xyz):
    nx, ny, nz = shape_xyz
    i, j, k = np.unravel_index(idx_flat, (nx, ny, nz), order="F")

    R = np.empty((idx_flat.size, 3, 3), dtype=np.float64)
    R[:, 0, 0] = fields["P11h"][i, j, k]; R[:, 0, 1] = fields["P12h"][i, j, k]; R[:, 0, 2] = fields["P13h"][i, j, k]
    R[:, 1, 0] = fields["P21h"][i, j, k]; R[:, 1, 1] = fields["P22h"][i, j, k]; R[:, 1, 2] = fields["P23h"][i, j, k]
    R[:, 2, 0] = fields["P31h"][i, j, k]; R[:, 2, 1] = fields["P32h"][i, j, k]; R[:, 2, 2] = fields["P33h"][i, j, k]
    return R


# ============================================================
# Project nearly-rotation matrices to SO(3) via SVD
# ============================================================
def project_to_so3_batch(R):
    U, _, Vt = np.linalg.svd(R, full_matrices=False)
    M = U @ Vt
    det = np.linalg.det(M)
    bad = det < 0.0
    if np.any(bad):
        U2 = U.copy()
        U2[bad, :, 2] *= -1.0
        M = U2 @ Vt
    return M


# ============================================================
# Cubic symmetry ops (24 proper rotations)
# ============================================================
def cubic_symmetry_ops():
    import itertools
    ops = []
    for p in itertools.permutations(range(3)):
        for s in itertools.product([-1.0, 1.0], repeat=3):
            M = np.zeros((3, 3), dtype=float)
            for i, j in enumerate(p):
                M[i, j] = s[i]
            if np.linalg.det(M) > 0.5:
                ops.append(M)
    if len(ops) != 24:
        raise RuntimeError(f"Expected 24 cubic ops, got {len(ops)}")
    return np.stack(ops, axis=0)


# ============================================================
# Stereographic projection + full stereonet
# ============================================================
def stereographic_xy(v):
    vx, vy, vz = v[..., 0], v[..., 1], v[..., 2]
    denom = 1.0 + vz
    denom = np.where(denom == 0, 1e-30, denom)
    return vx / denom, vy / denom


def draw_full_stereonet(ax, grid_deg=10):
    ax.set_aspect("equal", "box")
    ax.set_xlim(-1.02, 1.02)
    ax.set_ylim(-1.02, 1.02)
    ax.axis("off")

    t = np.linspace(0, 2*np.pi, 700)
    ax.plot(np.cos(t), np.sin(t), linewidth=1.2, color="k")

    deg = np.deg2rad

    # meridians
    for phi in range(0, 180, grid_deg):
        ph = deg(phi)
        th = np.linspace(0.0, np.pi/2, 300)
        v = np.stack([np.sin(th)*np.cos(ph),
                      np.sin(th)*np.sin(ph),
                      np.cos(th)], axis=1)
        x, y = stereographic_xy(v)
        ax.plot(x, y, linewidth=0.5, alpha=0.35, color="k")

    # parallels
    for theta in range(grid_deg, 90, grid_deg):
        th0 = deg(theta)
        ph = np.linspace(0, 2*np.pi, 420)
        v = np.stack([np.sin(th0)*np.cos(ph),
                      np.sin(th0)*np.sin(ph),
                      np.cos(th0)*np.ones_like(ph)], axis=1)
        x, y = stereographic_xy(v)
        ax.plot(x, y, linewidth=0.5, alpha=0.35, color="k")


# ============================================================
# Pole families under cubic symmetry (for PFs)
# ============================================================
def hkl_family_cubic(hkl, ops):
    hkl = np.array(hkl, dtype=float)
    fam = set()
    for S in ops:
        w = S @ hkl
        wi = tuple(int(round(x)) for x in w.tolist())

        # antipodal canonical sign (avoid +/- duplicates)
        a = list(wi)
        if a[0] != 0:
            sgn = 1 if a[0] > 0 else -1
        elif a[1] != 0:
            sgn = 1 if a[1] > 0 else -1
        else:
            sgn = 1 if a[2] > 0 else -1
        a = tuple(sgn * x for x in a)
        fam.add(a)

    fam = sorted(fam)
    out = []
    for t in fam:
        v = np.array(t, dtype=float)
        out.append(v / (np.linalg.norm(v) + 1e-30))
    return out


def pole_figure_dirs(R_sc, hkl, ops):
    """
    Return all pole directions n_s = R_cs * n_c in the sample frame
    (NO antipodal flipping — full sphere). Shape: (N_orient * k_family, 3).
    Used by both the scatter and the intensity (MRD) routines.
    """
    R_cs = np.transpose(R_sc, (0, 2, 1))
    fam = hkl_family_cubic(hkl, ops)

    dirs = []
    for n_c in fam:
        n_s = (R_cs @ n_c.reshape(1, 3, 1)).reshape(-1, 3)
        n_s /= (np.linalg.norm(n_s, axis=1, keepdims=True) + 1e-30)
        dirs.append(n_s)
    return np.concatenate(dirs, axis=0)


def pole_figure_xy(R_sc, hkl, ops):
    """
    Stereographic xy for scatter pole figures (upper hemisphere).
    Internally uses pole_figure_dirs and flips to z >= 0 via antipodal symmetry.
    """
    dirs = pole_figure_dirs(R_sc, hkl, ops).copy()
    flip = dirs[:, 2] < 0
    dirs[flip] *= -1.0
    x, y = stereographic_xy(dirs)
    return x, y


# ============================================================
# Pole figure INTENSITY (MRD) on a stereographic grid
# ============================================================
def pole_figure_intensity_mrd(dirs, n_grid=200, kernel_deg=7.5,
                              kernel_type="gauss"):
    """
    Compute pole figure intensity in multiples of random distribution (MRD)
    using a kernel density estimate on the sphere with antipodal symmetry.

    Parameters
    ----------
    dirs : (N, 3) ndarray
        Unit pole vectors in sample frame (NO flip — both hemispheres OK).
        Antipodal equivalence is enforced via |g . n|.
    n_grid : int
        Number of stereographic-grid points per axis.
    kernel_deg : float
        Half-width of the smoothing kernel, in degrees.
          - For kernel_type="cone": radius of the spherical cap (top-hat).
          - For kernel_type="gauss": Gaussian sigma in angular distance.
    kernel_type : "gauss" | "cone"
        Kernel shape.

    Returns
    -------
    Xg, Yg : (n_grid, n_grid) ndarray
        Stereographic grid coordinates (in unit disk).
    Z : np.ma.MaskedArray
        MRD values, masked outside the unit disk. MRD=1 corresponds to
        a uniform random texture.

    Notes
    -----
    Normalization derivation (for both kernels) is based on a uniform random
    distribution of N pole vectors on the full sphere (density N/4pi/sr).
    Using |g . n| (antipodal symmetry), the effective integrated kernel over
    the sphere is doubled, giving:
        cone:     E[count]  = N * (1 - cos(kernel_rad))
        gauss:    E[weight] = N * sigma^2                (narrow-kernel limit)
    so MRD is normalized accordingly.
    """
    # Stereographic grid (equal-angle Wulff convention)
    g1d = np.linspace(-1.0, 1.0, n_grid)
    Xg, Yg = np.meshgrid(g1d, g1d, indexing="xy")
    r2 = Xg * Xg + Yg * Yg
    grid_mask = r2 <= 1.0

    # Inverse stereographic projection -> unit vectors on upper hemisphere
    denom = 1.0 + r2
    Vx = 2.0 * Xg / denom
    Vy = 2.0 * Yg / denom
    Vz = (1.0 - r2) / denom
    grid_flat = np.stack([Vx.ravel(), Vy.ravel(), Vz.ravel()], axis=1)

    kernel_rad = np.deg2rad(kernel_deg)
    N = dirs.shape[0]

    # Chunk over grid rows to bound memory
    target_bytes = 2.5e8  # ~250 MB per chunk
    chunk = max(1, int(target_bytes / (8.0 * max(N, 1))))

    out_flat = np.zeros(grid_flat.shape[0], dtype=np.float64)

    if kernel_type == "cone":
        cos_thresh = np.cos(kernel_rad)
        for i in range(0, grid_flat.shape[0], chunk):
            sub = grid_flat[i:i + chunk]
            dots = sub @ dirs.T
            np.abs(dots, out=dots)
            out_flat[i:i + chunk] = np.sum(dots >= cos_thresh, axis=1)
        denom_norm = N * (1.0 - cos_thresh) + 1e-30
        mrd_flat = out_flat / denom_norm

    elif kernel_type == "gauss":
        sigma = kernel_rad
        inv2s2 = 1.0 / (2.0 * sigma * sigma)
        for i in range(0, grid_flat.shape[0], chunk):
            sub = grid_flat[i:i + chunk]
            dots = sub @ dirs.T
            np.abs(dots, out=dots)
            np.clip(dots, -1.0, 1.0, out=dots)
            d = np.arccos(dots)
            np.multiply(d, d, out=d)
            np.multiply(d, -inv2s2, out=d)
            np.exp(d, out=d)
            out_flat[i:i + chunk] = d.sum(axis=1)
        denom_norm = N * sigma * sigma + 1e-30
        mrd_flat = out_flat / denom_norm

    else:
        raise ValueError(f"Unknown kernel_type: {kernel_type!r} (use 'gauss' or 'cone').")

    Z = mrd_flat.reshape(Xg.shape)
    Z = np.ma.array(Z, mask=~grid_mask)
    return Xg, Yg, Z


# ============================================================
# IPF "standard triangle" mapping + MRD contours
# ============================================================
def ipf_triangle_xy_from_dirs(d_c):
    """
    d_c: (N,3) directions in crystal frame.

    Cubic direction reduction (directional + antipodal):
      - abs -> first octant
      - sort components descending: u>=v>=w>=0

    Standard triangle coordinates (straight edges):
      x = v/u
      y = w/u
    so triangle boundaries are: y=0, x=1, y=x.
    """
    v = np.abs(d_c)
    v.sort(axis=1)          # ascending
    v = v[:, ::-1]          # descending -> u,v,w
    u = v[:, 0]
    vv = v[:, 1]
    w = v[:, 2]

    u = np.where(u == 0.0, 1e-30, u)
    x = vv / u
    y = w / u

    x = np.clip(x, 0.0, 1.0)
    y = np.clip(y, 0.0, 1.0)
    return x, y


def histogram_mrd(x, y, Hr, bins):
    H, xedges, yedges = np.histogram2d(
        x, y, bins=bins, range=[[0.0, 1.0], [0.0, 1.0]]
    )

    eps = 1e-12
    p = H / (H.sum() + eps)
    pr = Hr / (Hr.sum() + eps)
    mrd = p / (pr + eps)

    xc = 0.5 * (xedges[:-1] + xedges[1:])
    yc = 0.5 * (yedges[:-1] + yedges[1:])
    X, Y = np.meshgrid(xc, yc, indexing="ij")

    # mask outside triangle y > x
    mrd = np.ma.array(mrd, mask=(Y > X))
    return X, Y, mrd


def plot_ipf_triangle(ax, X, Y, Zmrd, title, levels, show_filled=False, show_points=None):
    # Triangle boundary (straight edges): y=0, x=1, y=x
    ax.plot([0, 1], [0, 0], linewidth=1.6)
    ax.plot([1, 1], [0, 1], linewidth=1.6)
    ax.plot([0, 1], [0, 1], linewidth=1.6)

    if show_filled:
        ax.contourf(X, Y, Zmrd, levels=levels, alpha=0.35)

    cs = ax.contour(X, Y, Zmrd, levels=levels, linewidths=1.2)
    ax.clabel(cs, inline=True, fontsize=9, fmt="%g")

    # Corner labels
    ax.text(0.02, 0.02, "[001]", ha="left", va="bottom", fontsize=10)
    ax.text(0.98, 0.02, "[101]/[110]", ha="right", va="bottom", fontsize=10)
    ax.text(0.98, 0.98, "[111]", ha="right", va="top", fontsize=10)

    if show_points is not None:
        xp, yp = show_points
        inside = yp <= xp
        ax.scatter(xp[inside], yp[inside], s=2.0, alpha=0.25)

    ax.text(0.07, 0.90, title, fontsize=12, weight="bold")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal", "box")
    ax.set_xticks([])
    ax.set_yticks([])


# ============================================================
# Euler cube (Bunge) + cubic reduction
# ============================================================
def euler_bunge_from_g(g):
    g33 = np.clip(g[:, 2, 2], -1.0, 1.0)
    Phi = np.arccos(g33)
    sPhi = np.sin(Phi)

    phi1 = np.zeros_like(Phi)
    phi2 = np.zeros_like(Phi)

    eps = 1e-12
    ok = sPhi > eps

    phi1[ok] = np.arctan2(g[ok, 2, 0], -g[ok, 2, 1]) % (2*np.pi)
    phi2[ok] = np.arctan2(g[ok, 0, 2],  g[ok, 1, 2]) % (2*np.pi)

    sing = ~ok
    if np.any(sing):
        phi1[sing] = (np.arctan2(g[sing, 0, 1], g[sing, 0, 0]) % (2*np.pi))
        phi2[sing] = 0.0

    return phi1, Phi, phi2


def reduce_euler_to_cubic_cube(g, ops):
    # crystal symmetry: g' = g @ C
    cand = np.einsum("nij,kjl->knil", g, ops)  # (24,N,3,3)

    ph1_all = np.empty((24, g.shape[0]))
    PH_all  = np.empty((24, g.shape[0]))
    ph2_all = np.empty((24, g.shape[0]))

    for i in range(24):
        ph1, PH, ph2 = euler_bunge_from_g(cand[i])
        ph1_all[i], PH_all[i], ph2_all[i] = ph1, PH, ph2

    hi = 0.5*np.pi  # 90 deg
    inside = (ph1_all <= hi) & (PH_all <= hi) & (ph2_all <= hi)

    dist = np.maximum(ph1_all - hi, 0) + np.maximum(PH_all - hi, 0) + np.maximum(ph2_all - hi, 0)
    dist = np.where(inside, dist - 10.0, dist)

    best = np.argmin(dist, axis=0)
    ph1 = ph1_all[best, np.arange(g.shape[0])]
    PH  = PH_all[best,  np.arange(g.shape[0])]
    ph2 = ph2_all[best, np.arange(g.shape[0])]

    return np.rad2deg(ph1), np.rad2deg(PH), np.rad2deg(ph2)


# ============================================================
# Saving helper
# ============================================================
def save_figure(fig, outdir, prefix, tag, formats, dpi):
    if not outdir:
        return
    os.makedirs(outdir, exist_ok=True)
    for fmt in formats:
        path = os.path.join(outdir, f"{prefix}_{tag}.{fmt}")
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        print(f"Saved: {path}")


# ============================================================
# Colormap truncation
#   Returns a new colormap that uses only the [lo, hi] sub-range
#   of the input colormap. Useful for skipping the very dark
#   bottom of jet/rainbow so low MRD values don't appear as navy.
# ============================================================
def truncate_cmap(name, lo=0.0, hi=1.0, n=256):
    import matplotlib.pyplot as _plt
    from matplotlib.colors import LinearSegmentedColormap
    lo = float(max(0.0, min(1.0, lo)))
    hi = float(max(0.0, min(1.0, hi)))
    if lo >= hi:
        raise RuntimeError(f"--pf-cmap-min ({lo}) must be < --pf-cmap-max ({hi}).")
    base = _plt.get_cmap(name)
    colors = base(np.linspace(lo, hi, n))
    return LinearSegmentedColormap.from_list(f"{name}_trunc_{lo:.2f}_{hi:.2f}",
                                             colors, N=n)


# ============================================================
# Main
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("vtk", help="Path to fields_XXXXXX.vtk (ASCII STRUCTURED_POINTS)")
    ap.add_argument("--sense", choices=["s2c", "c2s"], default="s2c",
                    help="Matrix sense in file. s2c means v_c=R*v_s (your case).")

    ap.add_argument("--phi-thresh", type=float, default=0.5,
                    help="Use only voxels with phi >= this. Set <=0 to use all.")
    ap.add_argument("--max-ori", type=int, default=50000,
                    help="Max orientations to use (random downsample).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-project", action="store_true",
                    help="Skip SO(3) projection (faster; OK if matrices are already orthonormal).")

    # Stereonet grid spacing for PF/stereogram
    ap.add_argument("--grid-deg", type=int, default=10, help="Stereonet grid spacing in degrees.")

    # Build direction defaults (your case: ND=y)
    ap.add_argument("--nd", default="y", help="ND axis token (default y for build direction).")
    ap.add_argument("--rd", default="x", help="RD axis token (default x).")
    ap.add_argument("--td", default="z", help="TD axis token (default z).")

    # IPF contour options (MRD)
    ap.add_argument("--bins", type=int, default=120, help="Histogram bins per axis for IPF MRD.")
    ap.add_argument("--nrand", type=int, default=300000,
                    help="Random directions for MRD baseline (bigger -> smoother).")
    ap.add_argument("--levels", default="0.5,1,2,4",
                    help="IPF contour levels in MRD, comma-separated (e.g. 0.5,1,2,4).")
    ap.add_argument("--filled", action="store_true", help="Use filled contours under IPF lines.")
    ap.add_argument("--scatter", action="store_true", help="Overlay scatter points in IPFs.")

    # -------- Pole figure intensity (MRD) options --------
    ap.add_argument("--pf-grid", type=int, default=200,
                    help="Stereographic grid resolution (per axis) for PF intensity.")
    ap.add_argument("--pf-kernel", type=float, default=7.5,
                    help="Kernel half-width in degrees for PF intensity smoothing "
                         "(typical 5-15 deg).")
    ap.add_argument("--pf-kernel-type", choices=["gauss", "cone"], default="gauss",
                    help="Kernel shape for PF intensity. Default: gauss.")
    ap.add_argument("--pf-levels", default="0.5,1,1.5,2,2.5,3,3.5,4",
                    help="MRD contour levels for PF intensity (comma-separated). "
                         "Values outside [--pf-vmin, --pf-vmax] are clipped/extended.")
    ap.add_argument("--pf-vmin", type=float, default=0.0,
                    help="Lower limit of the PF intensity color scale (MRD). Default 0.")
    ap.add_argument("--pf-vmax", type=float, default=4.0,
                    help="Upper limit of the PF intensity color scale (MRD). "
                         "MRD values above this are saturated to the top color. Default 4.")
    ap.add_argument("--pf-cmap", default="jet",
                    help="Matplotlib colormap for PF intensity "
                         "(e.g. jet, turbo, rainbow, hsv, viridis, magma). Default: jet.")
    ap.add_argument("--pf-cmap-min", type=float, default=0.2,
                    help="Lower fraction of the colormap to use, in [0,1). "
                         "0.2 skips the bottom 20%% (removes the dark navy in jet). "
                         "Default: 0.2.")
    ap.add_argument("--pf-cmap-max", type=float, default=1.0,
                    help="Upper fraction of the colormap to use, in (0,1]. Default: 1.0.")
    ap.add_argument("--pf-no-scatter", action="store_true",
                    help="Skip the scatter pole-figure figure (keep only intensity).")
    ap.add_argument("--pf-no-intensity", action="store_true",
                    help="Skip the intensity pole-figure figure (keep only scatter).")
    # ----------------------------------------------------------

    # Save / run options
    ap.add_argument("--outdir", default="", help="If set, save figures to this directory.")
    ap.add_argument("--prefix", default="texture", help="Filename prefix for saved figures.")
    ap.add_argument("--formats", default="png",
                    help="Comma-separated formats to save (e.g. png,pdf,svg).")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--noshow", action="store_true", help="Do not open interactive windows.")

    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    want = ["phi",
            "P11h","P12h","P13h",
            "P21h","P22h","P23h",
            "P31h","P32h","P33h"]

    (nx, ny, nz), F = read_vtk_structured_points_ascii(args.vtk, want)

    # ----------------------------------------------------------
    # Voxel selection: phi threshold (WHOLE DOMAIN, no slicing)
    # ----------------------------------------------------------
    if args.phi_thresh > 0.0:
        mask = F["phi"] >= args.phi_thresh
        idx = np.flatnonzero(mask.ravel(order="F"))
    else:
        idx = np.arange(nx * ny * nz, dtype=np.int64)

    if idx.size == 0:
        raise RuntimeError("No voxels selected (phi-thresh too high?).")

    print(f"[info] {idx.size} voxels selected (whole domain, "
          f"{nx}x{ny}x{nz} = {nx*ny*nz} total)")

    if idx.size > args.max_ori:
        idx = rng.choice(idx, size=args.max_ori, replace=False)
        print(f"[info] downsampled to {args.max_ori} orientations")

    domain_label = f"whole domain ({nx}x{ny}x{nz})"

    # Build matrices
    R_data = build_R_from_fields(F, idx, (nx, ny, nz))

    # Interpret stored sense
    if args.sense == "s2c":
        R_sc = R_data
    else:
        R_sc = np.transpose(R_data, (0, 2, 1))

    if not args.no_project:
        R_sc = project_to_so3_batch(R_sc)

    R_cs = np.transpose(R_sc, (0, 2, 1))  # crystal->sample

    # Directions in sample frame (defaults: ND=y build, RD=x, TD=z)
    RD = parse_axis(args.rd)
    TD = parse_axis(args.td)
    ND = parse_axis(args.nd)

    formats = [s.strip() for s in args.formats.split(",") if s.strip()]

    # Cubic symmetry ops for PFs and Euler cube
    ops = cubic_symmetry_ops()

    hkls   = [(1,0,0), (1,1,0), (1,1,1)]
    titles = ["{100}", "{110}", "{111}"]

    # ============================================================
    # 1) Pole figures — SCATTER (full stereonet)
    # ============================================================
    if not args.pf_no_scatter:
        fig_pf, axs = plt.subplots(1, 3, figsize=(15, 5))
        for ax, hkl, title in zip(axs, hkls, titles):
            draw_full_stereonet(ax, grid_deg=args.grid_deg)
            x, y = pole_figure_xy(R_sc, hkl, ops)
            ax.scatter(x, y, s=1.0, alpha=0.35)
            ax.set_title(f"PF {title}")
        fig_pf.suptitle(f"Pole Figures — scatter (cubic, full net) — {domain_label}",
                        fontsize=11, y=0.99)
        fig_pf.tight_layout(rect=[0, 0, 1, 0.95])
        save_figure(fig_pf, args.outdir, args.prefix,
                    "pole_figures_scatter", formats, args.dpi)

    # ============================================================
    # 1b) Pole figures — INTENSITY (MRD contours, KDE on the sphere)
    # ============================================================
    if not args.pf_no_intensity:
        pf_levels_all = sorted(
            float(x.strip()) for x in args.pf_levels.split(",") if x.strip()
        )
        # Restrict levels to lie inside [vmin, vmax]; anything outside is
        # handled by `extend` (so peaks above vmax get the saturated top color).
        vmin, vmax = float(args.pf_vmin), float(args.pf_vmax)
        if vmin >= vmax:
            raise RuntimeError("--pf-vmin must be < --pf-vmax")
        pf_levels = [L for L in pf_levels_all if vmin <= L <= vmax]
        if len(pf_levels) < 2:
            # Fall back to a uniform 5-step ramp if the user-supplied levels
            # don't fit inside the [vmin, vmax] window.
            pf_levels = list(np.linspace(vmin, vmax, 5))

        # Build a (possibly truncated) colormap so the bottom of e.g. jet
        # doesn't show as a dark navy at low MRD.
        cmap_obj = truncate_cmap(args.pf_cmap,
                                 lo=args.pf_cmap_min,
                                 hi=args.pf_cmap_max)

        fig_pfi, axs_pfi = plt.subplots(1, 3, figsize=(16, 5.5))

        for ax, hkl, title in zip(axs_pfi, hkls, titles):
            ax.set_aspect("equal", "box")
            ax.set_xlim(-1.05, 1.05)
            ax.set_ylim(-1.05, 1.05)
            ax.axis("off")

            # Compute MRD grid
            dirs_pf = pole_figure_dirs(R_sc, hkl, ops)
            Xg, Yg, Zmrd = pole_figure_intensity_mrd(
                dirs_pf,
                n_grid=args.pf_grid,
                kernel_deg=args.pf_kernel,
                kernel_type=args.pf_kernel_type,
            )

            zmax = float(np.ma.max(Zmrd))
            zmin = float(np.ma.min(Zmrd))

            # Clip the data for display so the colormap really spans [vmin, vmax].
            # The actual underlying zmax/zmin are kept and shown in the title.
            Zdisp = np.ma.array(np.clip(Zmrd.filled(vmin), vmin, vmax),
                                mask=Zmrd.mask)

            # Filled colored contours (color scale locked to [vmin, vmax])
            cf = ax.contourf(Xg, Yg, Zdisp, levels=pf_levels,
                             cmap=cmap_obj, vmin=vmin, vmax=vmax,
                             extend="both")

            # Black contour lines with MRD labels
            cs = ax.contour(Xg, Yg, Zdisp, levels=pf_levels,
                            colors="k", linewidths=0.7, alpha=0.85)
            ax.clabel(cs, inline=True, fontsize=8, fmt="%g")

            # Re-draw stereonet on top so the unit-circle boundary
            # and a sparse polar grid remain readable over the colors.
            draw_full_stereonet(ax, grid_deg=args.grid_deg)

            # RD / TD axis tick labels (RD = x-axis of plot, TD = y-axis)
            ax.text( 1.06,  0.00, "RD", ha="left",   va="center", fontsize=10)
            ax.text( 0.00,  1.06, "TD", ha="center", va="bottom", fontsize=10)

            ax.set_title(f"PF {title}   max MRD = {zmax:.2f}   "
                         f"min = {zmin:.2f}", fontsize=10)

            cbar = fig_pfi.colorbar(cf, ax=ax, shrink=0.78, pad=0.04, label="MRD")
            cbar.set_ticks(pf_levels)

        fig_pfi.suptitle(
            f"Pole Figure Intensity (cubic, MRD; "
            f"{args.pf_kernel_type} kernel, half-width = {args.pf_kernel}°; "
            f"cmap {args.pf_cmap}[{args.pf_cmap_min:g}:{args.pf_cmap_max:g}], "
            f"scale {vmin:g}–{vmax:g}) — {domain_label}",
            fontsize=11, y=0.995
        )
        fig_pfi.tight_layout(rect=[0, 0, 1, 0.94])
        save_figure(fig_pfi, args.outdir, args.prefix,
                    "pole_figures_intensity", formats, args.dpi)

    # ============================================================
    # 2) Full stereonet stereogram: crystal [001] direction in sample
    # ============================================================
    fig_st, ax = plt.subplots(1, 1, figsize=(6, 6))
    draw_full_stereonet(ax, grid_deg=args.grid_deg)
    e3c = np.array([0.0, 0.0, 1.0])  # crystal [001]
    v_s = (R_cs @ e3c.reshape(1,3,1)).reshape(-1,3)
    v_s /= (np.linalg.norm(v_s, axis=1, keepdims=True) + 1e-30)
    flip = v_s[:, 2] < 0
    v_s[flip] *= -1.0
    xs, ys = stereographic_xy(v_s)
    ax.scatter(xs, ys, s=1.0, alpha=0.35)
    ax.set_title(f"Stereogram: crystal [001] in sample — {domain_label}")
    fig_st.tight_layout()
    save_figure(fig_st, args.outdir, args.prefix, "stereogram_001_fullnet", formats, args.dpi)

    # ============================================================
    # 3) IPF ND/RD/TD (standard triangle) with MRD contours
    # ============================================================
    # Random baseline (uniform on sphere)
    g = rng.normal(size=(args.nrand, 3))
    g /= (np.linalg.norm(g, axis=1, keepdims=True) + 1e-30)
    xr, yr = ipf_triangle_xy_from_dirs(g)
    Hr, _, _ = np.histogram2d(xr, yr, bins=args.bins, range=[[0.0, 1.0], [0.0, 1.0]])

    levels = [float(x.strip()) for x in args.levels.split(",") if x.strip()]
    levels = sorted(levels)

    def ipf_grid_for_dir(d_s):
        d_c = (R_sc @ d_s.reshape(1, 3, 1)).reshape(-1, 3)
        d_c /= (np.linalg.norm(d_c, axis=1, keepdims=True) + 1e-30)
        x, y = ipf_triangle_xy_from_dirs(d_c)
        X, Y, Zmrd = histogram_mrd(x, y, Hr, bins=args.bins)
        return (x, y), (X, Y, Zmrd)

    pts_ND, grid_ND = ipf_grid_for_dir(ND)
    pts_RD, grid_RD = ipf_grid_for_dir(RD)
    pts_TD, grid_TD = ipf_grid_for_dir(TD)

    fig_ipf, axs = plt.subplots(1, 3, figsize=(14.5, 5.0))
    plot_ipf_triangle(axs[0], *grid_ND, title="ND", levels=levels,
                      show_filled=args.filled, show_points=(pts_ND if args.scatter else None))
    plot_ipf_triangle(axs[1], *grid_RD, title="RD", levels=levels,
                      show_filled=args.filled, show_points=(pts_RD if args.scatter else None))
    plot_ipf_triangle(axs[2], *grid_TD, title="TD", levels=levels,
                      show_filled=args.filled, show_points=(pts_TD if args.scatter else None))

    fig_ipf.suptitle(
        f"IPF (cubic, MRD). RD={args.rd}, TD={args.td}, ND={args.nd} — {domain_label}",
        y=0.98, fontsize=12
    )
    fig_ipf.tight_layout(rect=[0, 0, 1, 0.95])
    save_figure(fig_ipf, args.outdir, args.prefix,
                f"ipf_contours_RD{args.rd}_TD{args.td}_ND{args.nd}", formats, args.dpi)

    # ============================================================
    # 4) Euler cube scatter (cubic-reduced)
    # ============================================================
    gmat = R_cs  # crystal->sample for Euler (Bunge)
    e1, e2, e3 = reduce_euler_to_cubic_cube(gmat, ops)

    fig_eu = plt.figure(figsize=(7, 6))
    ax3 = fig_eu.add_subplot(111, projection="3d")
    ax3.scatter(e1, e2, e3, s=2.0, alpha=0.35)
    ax3.set_xlabel(r"$\phi_1$ (deg)")
    ax3.set_ylabel(r"$\Phi$ (deg)")
    ax3.set_zlabel(r"$\phi_2$ (deg)")
    ax3.set_xlim(0, 90); ax3.set_ylim(0, 90); ax3.set_zlim(0, 90)
    ax3.set_title(f"Euler cube scatter (cubic reduced) — {domain_label}")
    fig_eu.tight_layout()
    save_figure(fig_eu, args.outdir, args.prefix, "euler_cube", formats, args.dpi)

    if not args.noshow:
        plt.show()


if __name__ == "__main__":
    main()