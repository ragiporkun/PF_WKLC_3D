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
    ax.plot(np.cos(t), np.sin(t), linewidth=1.2)

    deg = np.deg2rad

    # meridians
    for phi in range(0, 180, grid_deg):
        ph = deg(phi)
        th = np.linspace(0.0, np.pi/2, 300)
        v = np.stack([np.sin(th)*np.cos(ph),
                      np.sin(th)*np.sin(ph),
                      np.cos(th)], axis=1)
        x, y = stereographic_xy(v)
        ax.plot(x, y, linewidth=0.5, alpha=0.35)

    # parallels
    for theta in range(grid_deg, 90, grid_deg):
        th0 = deg(theta)
        ph = np.linspace(0, 2*np.pi, 420)
        v = np.stack([np.sin(th0)*np.cos(ph),
                      np.sin(th0)*np.sin(ph),
                      np.cos(th0)*np.ones_like(ph)], axis=1)
        x, y = stereographic_xy(v)
        ax.plot(x, y, linewidth=0.5, alpha=0.35)


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


def pole_figure_xy(R_sc, hkl, ops):
    # n_s = R_cs * n_c = (R_sc^T) * n_c
    R_cs = np.transpose(R_sc, (0, 2, 1))
    fam = hkl_family_cubic(hkl, ops)

    xs, ys = [], []
    for n_c in fam:
        n_s = (R_cs @ n_c.reshape(1, 3, 1)).reshape(-1, 3)
        n_s /= (np.linalg.norm(n_s, axis=1, keepdims=True) + 1e-30)
        # antipodal to upper hemisphere
        flip = n_s[:, 2] < 0
        n_s[flip] *= -1.0
        x, y = stereographic_xy(n_s)
        xs.append(x); ys.append(y)

    return np.concatenate(xs), np.concatenate(ys)


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
# Main
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("vtk", help="Path to fields_XXXXXX.vtk (ASCII STRUCTURED_POINTS)")
    ap.add_argument("--sense", choices=["s2c", "c2s"], default="s2c",
                    help="Matrix sense in file. s2c means v_c=R*v_s.")

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

    # ---- Slice selection ----
    ap.add_argument("--slice", choices=["x", "y", "z"], default="x",
                    help="Extract a 2D slice normal to this axis. "
                         "Default is 'x' to take the y-z plane.")
    ap.add_argument("--slice-index", type=int, default=None,
                    help="Explicit voxel index along the slice axis (0-based). "
                         "Default: midplane (N//2), which matches the origin at nx*dx/2.")

    # IPF contour options (MRD)
    ap.add_argument("--bins", type=int, default=120, help="Histogram bins per axis for IPF MRD.")
    ap.add_argument("--nrand", type=int, default=300000,
                    help="Random directions for MRD baseline (bigger -> smoother).")
    ap.add_argument("--levels", default="0.5,1,2,4",
                    help="IPF contour levels in MRD, comma-separated (e.g. 0.5,1,2,4).")
    ap.add_argument("--filled", action="store_true", help="Use filled contours under IPF lines.")
    ap.add_argument("--scatter", action="store_true", help="Overlay scatter points in IPFs.")

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
    # Voxel selection: phi threshold
    # ----------------------------------------------------------
    if args.phi_thresh > 0.0:
        mask = F["phi"] >= args.phi_thresh
    else:
        mask = np.ones((nx, ny, nz), dtype=bool)

    # ----------------------------------------------------------
    # Slice selection: restrict to a single plane
    # ----------------------------------------------------------
    slice_label = "full domain"
    if args.slice is not None:
        axis_map = {"x": 0, "y": 1, "z": 2}
        dim_sizes = {"x": nx, "y": ny, "z": nz}
        sl_ax = axis_map[args.slice]
        sl_n  = dim_sizes[args.slice]

        if args.slice_index is not None:
            sl_k = args.slice_index
        else:
            sl_k = sl_n // 2  # midplane

        if sl_k < 0 or sl_k >= sl_n:
            raise RuntimeError(
                f"--slice-index {sl_k} out of range for axis {args.slice} "
                f"(N{args.slice}={sl_n}, valid 0..{sl_n-1})."
            )

        # Build a mask for just this slice
        slice_mask = np.zeros((nx, ny, nz), dtype=bool)
        if args.slice == "x":
            slice_mask[sl_k, :, :] = True
        elif args.slice == "y":
            slice_mask[:, sl_k, :] = True
        else:  # z
            slice_mask[:, :, sl_k] = True

        mask = mask & slice_mask

        plane_labels = {"x": "y-z", "y": "x-z", "z": "x-y"}
        slice_label = (f"{plane_labels[args.slice]} plane at "
                       f"{args.slice}={sl_k}/{sl_n} (index {sl_k})")
        print(f"[slice] Extracting {slice_label}")

    idx = np.flatnonzero(mask.ravel(order="F"))

    if idx.size == 0:
        raise RuntimeError("No voxels selected (phi-thresh too high or empty slice?).")

    print(f"[info] {idx.size} voxels selected ({slice_label})")

    if idx.size > args.max_ori:
        idx = rng.choice(idx, size=args.max_ori, replace=False)

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

    # Directions in sample frame
    RD = parse_axis(args.rd)
    TD = parse_axis(args.td)
    ND = parse_axis(args.nd)

    formats = [s.strip() for s in args.formats.split(",") if s.strip()]

    # Cubic symmetry ops
    ops = cubic_symmetry_ops()

    if args.slice is not None:
        file_tag = f"slice_{args.slice}{sl_k}"
    else:
        file_tag = "full"

    # ============================================================
    # 1) Pole figures (full stereonet)
    # ============================================================
    hkls = [(0,0,1), (1,1,0), (1,1,1)]
    titles = ["{001}", "{110}", "{111}"]

    fig_pf, axs = plt.subplots(1, 3, figsize=(15, 5))
    for ax, hkl, title in zip(axs, hkls, titles):
        draw_full_stereonet(ax, grid_deg=args.grid_deg)
        x, y = pole_figure_xy(R_sc, hkl, ops)
        ax.scatter(x, y, s=1.0, alpha=0.35)
        ax.set_title(f"PF {title}")
    fig_pf.suptitle(f"Pole Figures (cubic, full net) — {slice_label}", fontsize=11, y=0.99)
    fig_pf.tight_layout(rect=[0, 0, 1, 0.95])
    save_figure(fig_pf, args.outdir, args.prefix, f"pole_figures_fullnet_{file_tag}", formats, args.dpi)

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
    ax.set_title(f"Stereogram: crystal [001] in sample — {slice_label}")
    fig_st.tight_layout()
    save_figure(fig_st, args.outdir, args.prefix, f"stereogram_001_fullnet_{file_tag}", formats, args.dpi)

    # ============================================================
    # 3) IPF ND/RD/TD (standard triangle) with MRD contours
    # ============================================================
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
        f"IPF (cubic, MRD). RD={args.rd}, TD={args.td}, ND={args.nd} — {slice_label}",
        y=0.98, fontsize=12
    )
    fig_ipf.tight_layout(rect=[0, 0, 1, 0.95])
    save_figure(fig_ipf, args.outdir, args.prefix,
                f"ipf_contours_RD{args.rd}_TD{args.td}_ND{args.nd}_{file_tag}", formats, args.dpi)

    # ============================================================
    # 4) Euler cube scatter (cubic-reduced)
    # ============================================================
    gmat = R_cs
    e1, e2, e3 = reduce_euler_to_cubic_cube(gmat, ops)

    fig_eu = plt.figure(figsize=(7, 6))
    ax3 = fig_eu.add_subplot(111, projection="3d")
    ax3.scatter(e1, e2, e3, s=2.0, alpha=0.35)
    ax3.set_xlabel(r"$\phi_1$ (deg)")
    ax3.set_ylabel(r"$\Phi$ (deg)")
    ax3.set_zlabel(r"$\phi_2$ (deg)")
    ax3.set_xlim(0, 90); ax3.set_ylim(0, 90); ax3.set_zlim(0, 90)
    ax3.set_title(f"Euler cube (cubic reduced) — {slice_label}")
    fig_eu.tight_layout()
    save_figure(fig_eu, args.outdir, args.prefix, f"euler_cube_{file_tag}", formats, args.dpi)

    if not args.noshow:
        plt.show()


if __name__ == "__main__":
    main()