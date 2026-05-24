"""
Texture + microstructure analysis of fields_025000.vtk.

Corrections vs. previous version
--------------------------------
1. Grain segmentation now uses the proper *cubic disorientation* (min over the
   24x24 symmetry pairs) instead of the raw matrix angle. Without this, two
   voxels inside a single physical grain can be separated by 60+ degrees of
   "apparent" rotation just because the simulation's P-field landed in
   different symmetry variants on either side of a smooth gradient. The 5 deg
   threshold then cleaved every such crossing, producing the 912-grain
   oversegmentation (vs. the 180 seeds).
2. Union-Find replaced by scipy.sparse.csgraph.connected_components, which
   removes a ~1e6-call Python loop and is roughly two orders of magnitude
   faster.
3. Added a diagnostic histogram of neighbour angles (raw vs disorientation),
   so the symmetry artefact is visible directly.

Outputs (in /mnt/user-data/outputs):
  pole_figures.png
  grain_distributions.png
  neighbour_angles_diagnostic.png
  table2.csv, table2.md
  summary.json
"""
import json
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from scipy.ndimage import gaussian_filter
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial.transform import Rotation as Rot

t0 = time.time()
OUT = Path("outputs")
OUT.mkdir(parents=True, exist_ok=True)
VTK = "./VTK_P2000_v1_180seeds/fields_025000.vtk"
RNG = np.random.default_rng(0)

# Segmentation threshold in degrees on the cubic disorientation. 5 deg is the
# usual "same grain" cutoff for low-angle boundaries; raise to 10-15 deg if you
# want HAGB-based grain definition.
GRAIN_THRESHOLD_DEG = 5.0


# ----------------------------------------------------------------------
# 1. Load the VTK and assemble the rotation field
# ----------------------------------------------------------------------
print("[1] Reading VTK ...")
mesh = pv.read(VTK)
nx, ny, nz = mesh.dimensions          # (250, 50, 50) — # of points per axis
spacing = np.array(mesh.spacing)
phi = np.asarray(mesh.point_data["phi"]).reshape(nz, ny, nx)  # vtk order: x fastest

P_flat = np.stack([mesh.point_data[f"P{i}{j}h"] for i in (1, 2, 3) for j in (1, 2, 3)],
                  axis=1).reshape(-1, 3, 3)
solid_mask_flat = phi.reshape(-1) > 0.5
P_solid = P_flat[solid_mask_flat].astype(np.float64)

# Project to the nearest proper rotation via SVD
U, _, Vt = np.linalg.svd(P_solid)
det = np.linalg.det(U @ Vt)
Vt[..., -1, :] *= np.where(det < 0, -1.0, 1.0)[:, None]
P_solid = U @ Vt
Nvox = P_solid.shape[0]
print(f"    grid {nx}x{ny}x{nz}, solid voxels = {Nvox:,}")


# ----------------------------------------------------------------------
# 2. Cubic point-group rotation operators (24 proper rotations of m-3m)
# ----------------------------------------------------------------------
def cubic_symmetry_ops():
    ops = [np.eye(3)]
    for axis in np.eye(3):                                       # 4-fold about x,y,z
        for k in (1, 2, 3):
            ops.append(Rot.from_rotvec(axis * np.pi / 2 * k).as_matrix())
    for axis in [[1, 1, 1], [-1, 1, 1], [1, -1, 1], [1, 1, -1]]: # 3-fold about <111>
        a = np.array(axis) / np.sqrt(3)
        for k in (1, 2):
            ops.append(Rot.from_rotvec(a * 2 * np.pi / 3 * k).as_matrix())
    for axis in [[1, 1, 0], [1, -1, 0], [1, 0, 1],               # 2-fold <110>
                 [1, 0, -1], [0, 1, 1], [0, 1, -1]]:
        a = np.array(axis) / np.sqrt(2)
        ops.append(Rot.from_rotvec(a * np.pi).as_matrix())
    ops = np.stack(ops)
    assert ops.shape == (24, 3, 3)
    assert np.allclose(np.linalg.det(ops), 1.0)
    return ops

SYM = cubic_symmetry_ops()                          # (24,3,3)


# ----------------------------------------------------------------------
# Shared kernel: cubic disorientation angle (deg) between paired rotations.
# Used both for sampled-pair statistics (section 4) and grain segmentation
# (section 6). Batched to keep the (n, 24, 24) intermediate tensor bounded.
# ----------------------------------------------------------------------
def cubic_disorientation_angles(A, B, SYM=SYM, batch=20_000):
    """
    For each i, returns the cubic disorientation angle in degrees between
    rotation matrices A[i] and B[i]:  min over (Sa, Sb) of angle(Sa @ A^T B @ Sb).
    """
    A = np.asarray(A); B = np.asarray(B)
    out = np.empty(len(A))
    for s in range(0, len(A), batch):
        a = A[s:s+batch]; b = B[s:s+batch]
        rel = np.einsum("nji,njk->nik", a, b)              # (n,3,3) = a^T b
        best = np.full(len(rel), -1.0)
        for Sa in SYM:                                     # 24 outer
            Sa_rel = Sa @ rel                              # (n,3,3)
            # trace(Sa_rel @ Sb) for every Sb in one shot:
            tr = np.einsum("nij,bji->nb", Sa_rel, SYM)     # (n,24)
            np.maximum(best, tr.max(axis=1), out=best)
        out[s:s+batch] = np.degrees(np.arccos(np.clip((best - 1) / 2, -1, 1)))
    return out


# ----------------------------------------------------------------------
# 3. ODF on Bunge Euler grid -> J-index
# ----------------------------------------------------------------------
print("[2] Computing J-index ...")
nb1, nbP, nb2 = 72, 36, 72
step = 5.0
counts = np.zeros((nb1, nbP, nb2), dtype=np.float64)

BATCH = 40_000
for start in range(0, Nvox, BATCH):
    chunk = P_solid[start:start + BATCH]
    sym_chunk = np.einsum("nij,sjk->nsik", chunk, SYM).reshape(-1, 3, 3)
    eul = Rot.from_matrix(sym_chunk).as_euler("ZXZ", degrees=True)
    eul[:, 0] %= 360.0
    eul[:, 2] %= 360.0
    i1 = np.clip((eul[:, 0] / step).astype(int), 0, nb1 - 1)
    iP = np.clip((eul[:, 1] / step).astype(int), 0, nbP - 1)
    i2 = np.clip((eul[:, 2] / step).astype(int), 0, nb2 - 1)
    flat_idx = (i1 * nbP + iP) * nb2 + i2
    counts += np.bincount(flat_idx, minlength=nb1 * nbP * nb2).reshape(nb1, nbP, nb2)

Phi_centers = np.deg2rad((np.arange(nbP) + 0.5) * step)
dphi = np.deg2rad(step)
dV = (np.sin(Phi_centers) * dphi**3 / (8 * np.pi**2))
dV_grid = np.broadcast_to(dV[None, :, None], counts.shape)
total_weight = counts.sum()
f = counts / (total_weight * dV_grid)
J_index = float(np.sum(f**2 * dV_grid))
print(f"    J = {J_index:.3f}  (random = 1)")


# ----------------------------------------------------------------------
# 4. M-index (Mackenzie reference + sampled disorientations)
# ----------------------------------------------------------------------
print("[3] Computing M-index ...")
n_pairs = 100_000
i = RNG.integers(0, Nvox, size=n_pairs)
j = RNG.integers(0, Nvox, size=n_pairs)
diso = cubic_disorientation_angles(P_solid[i], P_solid[j])

n_ref = 200_000
Ra = Rot.random(n_ref, random_state=RNG).as_matrix()
Rb = Rot.random(n_ref, random_state=RNG).as_matrix()
ref = cubic_disorientation_angles(Ra, Rb)

bins = np.arange(0, 63.0, 1.0)
h_data, _ = np.histogram(diso, bins=bins, density=True)
h_ref,  _ = np.histogram(ref,  bins=bins, density=True)
M_index = float(0.5 * np.sum(np.abs(h_ref - h_data)) * (bins[1] - bins[0]))
print(f"    M = {M_index:.3f}  (random = 0, single-crystal = 1)")


# ----------------------------------------------------------------------
# 5. Pole figures (Lambert equal-area, upper hemisphere)
# ----------------------------------------------------------------------
print("[4] Computing pole figures ...")
def crystal_directions(hkl):
    h, k, l = hkl
    seen = set()
    out = []
    for sgn in np.array(np.meshgrid([1, -1], [1, -1], [1, -1])).T.reshape(-1, 3):
        for perm in [(0,1,2),(0,2,1),(1,0,2),(1,2,0),(2,0,1),(2,1,0)]:
            v = np.array([h, k, l])[list(perm)] * sgn
            key = tuple(v)
            if key in seen or tuple(-v) in seen:
                continue
            seen.add(key)
            out.append(v / np.linalg.norm(v))
    return np.array(out)

def pole_figure(P, hkl, n_grid=181, smooth_deg=5.0):
    dirs = crystal_directions(hkl)
    y = np.einsum("nij,mj->nmi", P, dirs).reshape(-1, 3)
    flip = y[:, 2] < 0
    y[flip] *= -1
    r_factor = np.sqrt(2.0 / (1.0 + y[:, 2]))
    xp = y[:, 0] * r_factor
    yp = y[:, 1] * r_factor
    R = np.sqrt(2.0)
    H, xe, ye = np.histogram2d(xp, yp, bins=n_grid, range=[[-R, R], [-R, R]])
    bin_area = (2 * R / n_grid) ** 2
    total = H.sum()
    mrd = (H / total) / (bin_area / (2 * np.pi))
    sigma_px = (smooth_deg / 90.0) * R / (2 * R / n_grid)
    mrd_smooth = gaussian_filter(mrd, sigma=sigma_px, mode="constant", cval=0.0)
    xs, ys = np.meshgrid((xe[:-1] + xe[1:]) / 2, (ye[:-1] + ye[1:]) / 2, indexing="ij")
    inside = (xs**2 + ys**2) <= 2.0
    mrd_smooth = np.where(inside, mrd_smooth, np.nan)
    return mrd_smooth, R, np.nanmax(mrd_smooth)

pfs = {}
for hkl, label in [((1,0,0), "100"), ((1,1,0), "110"), ((1,1,1), "111")]:
    mrd, R, peak = pole_figure(P_solid, hkl)
    pfs[label] = {"mrd": mrd, "R": R, "peak": float(peak)}
    print(f"    {{{label}}} peak m.r.d. = {peak:.2f}")

def plot_pf(label, payload, ax, exp_max=None):
    mrd, R, peak = payload["mrd"], payload["R"], payload["peak"]
    im = ax.imshow(mrd.T, extent=[-R, R, -R, R], origin="lower",
                   cmap="viridis", vmin=0, vmax=max(peak, 1.0))
    ax.add_patch(plt.Circle((0, 0), R, fill=False, color="k", lw=1))
    ax.set_xlim(-R*1.05, R*1.05); ax.set_ylim(-R*1.05, R*1.05)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    title = f"{{{label}}}  peak = {peak:.2f} m.r.d."
    if exp_max is not None:
        title += f"\n(exp. {exp_max})"
    ax.set_title(title, fontsize=10)
    return im

exp_pf = {"100": 4.0, "110": 6.5, "111": 4.8}
fig, axes = plt.subplots(1, 3, figsize=(11, 4.2))
for ax, lab in zip(axes, ["100", "110", "111"]):
    im = plot_pf(lab, pfs[lab], ax, exp_pf[lab])
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="m.r.d.")
fig.suptitle("Simulated pole figures (Lambert equal-area, upper hemisphere)", y=1.02)
fig.tight_layout()
fig.savefig(OUT / "pole_figures.png", dpi=160, bbox_inches="tight")
plt.close(fig)


# ----------------------------------------------------------------------
# 6. Grain segmentation: 6-neighbour graph, cubic disorientation < threshold
# ----------------------------------------------------------------------
print(f"[5] Segmenting grains (cubic disorientation < {GRAIN_THRESHOLD_DEG} deg) ...")

# Reshape P over the 3-D grid for easy neighbour indexing.
# VTK ASCII point-data ordering is x fastest, then y, then z, so reshape is
# (nz, ny, nx, 3, 3); transpose to (nx, ny, nz, 3, 3) for intuitive axes.
P_grid = P_flat.reshape(nz, ny, nx, 3, 3).transpose(2, 1, 0, 3, 4).copy()

# SVD-project the entire grid to proper rotations (cheap, vectorised)
U, _, Vt = np.linalg.svd(P_grid.reshape(-1, 3, 3))
det = np.linalg.det(U @ Vt)
Vt[..., -1, :] *= np.where(det < 0, -1.0, 1.0)[:, None]
P_grid = (U @ Vt).reshape(nx, ny, nz, 3, 3)

solid = (phi.transpose(2, 1, 0) > 0.5)                # (nx,ny,nz)
N = nx * ny * nz
flat_index = np.arange(N).reshape(nx, ny, nz)

# Raw no-symmetry angle, for the diagnostic plot only
def raw_edge_angles(A, B):
    rel = np.einsum("nji,njk->nik", A, B)
    tr = rel[:, 0, 0] + rel[:, 1, 1] + rel[:, 2, 2]
    return np.degrees(np.arccos(np.clip((tr - 1) / 2, -1, 1)))

# Build edges along each of the three axes
edge_rows, edge_cols = [], []
raw_angle_samples, diso_angle_samples = [], []   # for the diagnostic plot

for axis in range(3):
    src = [slice(None)] * 3
    dst = [slice(None)] * 3
    src[axis] = slice(0, -1)
    dst[axis] = slice(1, None)
    src = tuple(src); dst = tuple(dst)

    s_mask = solid[src] & solid[dst]
    if not s_mask.any():
        continue

    A = P_grid[src][s_mask]
    B = P_grid[dst][s_mask]
    idx_a = flat_index[src][s_mask]
    idx_b = flat_index[dst][s_mask]

    # Cubic disorientation for the segmentation decision
    diso = cubic_disorientation_angles(A, B)
    # Raw angle for the diagnostic (cheap, no symmetry)
    raw  = raw_edge_angles(A, B)

    keep = diso < GRAIN_THRESHOLD_DEG
    edge_rows.append(idx_a[keep])
    edge_cols.append(idx_b[keep])

    raw_angle_samples.append(raw)
    diso_angle_samples.append(diso)

edge_rows = np.concatenate(edge_rows) if edge_rows else np.array([], dtype=np.int64)
edge_cols = np.concatenate(edge_cols) if edge_cols else np.array([], dtype=np.int64)
raw_all  = np.concatenate(raw_angle_samples)
diso_all = np.concatenate(diso_angle_samples)
print(f"    solid-solid neighbour edges: {len(raw_all):,}")
print(f"    fraction with raw angle > 5 deg : {(raw_all  > 5).mean():.4f}")
print(f"    fraction with diso  angle > 5 deg : {(diso_all > 5).mean():.4f}")

# Connected components via sparse graph (symmetric)
data = np.ones(len(edge_rows), dtype=bool)
G = coo_matrix((data, (edge_rows, edge_cols)), shape=(N, N))
ncc, labels = connected_components(G, directed=False)

# Mark liquid voxels as label = -1
labels = labels.astype(np.int64)
labels[~solid.reshape(-1)] = -1

# Grain stats (drop micro-fragments < 4 voxels, as before)
solid_flat_idx = np.where(solid.reshape(-1))[0]
solid_labels = labels[solid_flat_idx]
unique_lbl, sizes = np.unique(solid_labels, return_counts=True)

coords = np.stack(np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij"),
                  axis=-1).reshape(-1, 3)
coords_phys = coords * spacing

ar_list, vol_list = [], []
for lbl, sz in zip(unique_lbl, sizes):
    if sz < 4:
        continue
    pts = coords_phys[solid_flat_idx[solid_labels == lbl]]
    c = pts.mean(axis=0)
    cov = np.cov((pts - c).T)
    if cov.ndim == 0:        # single-point cluster after filtering
        continue
    eig = np.linalg.eigvalsh(cov)
    eig = np.clip(eig, 1e-12, None)
    ar = float(np.sqrt(eig.max() / eig.min()))
    ar_list.append(ar)
    vol_list.append(int(sz))

vol_arr = np.array(vol_list)
ar_arr  = np.array(ar_list)
diam_orig = (vol_arr * float(np.prod(spacing))) ** (1/3)
print(f"    grains found: {len(vol_arr)}   "
      f"size: median={np.median(vol_arr):.0f} vox, max={vol_arr.max()} vox   "
      f"AR median={np.median(ar_arr):.2f}")


# ----------------------------------------------------------------------
# 7. Diagnostic plot: raw vs cubic-disorientation neighbour angles
# ----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(raw_all,  bins=180, range=(0, 90), histtype="step",
        label=f"raw angle (no symmetry), >5 deg: {(raw_all > 5).mean()*100:.2f}%",
        color="#ef4444", lw=1.4)
ax.hist(diso_all, bins=180, range=(0, 90), histtype="step",
        label=f"cubic disorientation, >5 deg: {(diso_all > 5).mean()*100:.2f}%",
        color="#3b82f6", lw=1.4)
ax.axvline(GRAIN_THRESHOLD_DEG, color="k", ls="--", lw=1, label=f"{GRAIN_THRESHOLD_DEG} deg threshold")
ax.set_yscale("log")
ax.set_xlabel("Neighbour angle (deg)")
ax.set_ylabel("Edge count (log)")
ax.set_title("Solid-solid neighbour angle distribution")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "neighbour_angles_diagnostic.png", dpi=160, bbox_inches="tight")
plt.close(fig)


# ----------------------------------------------------------------------
# 8. Plot grain distributions
# ----------------------------------------------------------------------
fig, axs = plt.subplots(1, 2, figsize=(10, 4))
axs[0].hist(diam_orig, bins=30, color="#3b82f6", edgecolor="k", alpha=0.85)
axs[0].set_xlabel("Equivalent grain diameter (sim. units)")
axs[0].set_ylabel("Count")
axs[0].set_title(f"Grain size, N = {len(vol_arr)}")
axs[0].axvline(np.median(diam_orig), color="r", ls="--", lw=1,
               label=f"median = {np.median(diam_orig):.2f}")
axs[0].legend()

axs[1].hist(ar_arr, bins=30, color="#10b981", edgecolor="k", alpha=0.85)
axs[1].set_xlabel("Aspect ratio (sqrt(lambda_max / lambda_min))")
axs[1].set_ylabel("Count")
axs[1].set_title(f"Aspect ratio, median = {np.median(ar_arr):.2f}")
axs[1].axvline(np.median(ar_arr), color="r", ls="--", lw=1)
fig.tight_layout()
fig.savefig(OUT / "grain_distributions.png", dpi=160, bbox_inches="tight")
plt.close(fig)


# ----------------------------------------------------------------------
# 9. Table 2: simulated vs experimental
# ----------------------------------------------------------------------
exp = {
    "{100} peak m.r.d.": 4.0,
    "{110} peak m.r.d.": 6.5,
    "{111} peak m.r.d.": 4.8,
    "{110} max m.r.d. (Fig. 1h)": 9.5,
}
sim = {
    "{100} peak m.r.d.": pfs["100"]["peak"],
    "{110} peak m.r.d.": pfs["110"]["peak"],
    "{111} peak m.r.d.": pfs["111"]["peak"],
    "{110} max m.r.d. (Fig. 1h)": pfs["110"]["peak"],
    "Texture J-index": J_index,
    "Texture M-index": M_index,
    f"N grains (<{GRAIN_THRESHOLD_DEG:g} deg)": int(len(vol_arr)),
    "Median grain size (voxels)": float(np.median(vol_arr)),
    "Median equiv. diameter (sim units)": float(np.median(diam_orig)),
    "Median aspect ratio": float(np.median(ar_arr)),
}

rows = []
for k, v_sim in sim.items():
    v_exp = exp.get(k, "—")
    rows.append((k, v_sim, v_exp))

with open(OUT / "table2.csv", "w") as f:
    f.write("metric,simulated,experimental\n")
    for k, s, e in rows:
        s_str = f"{s:.3f}" if isinstance(s, float) else str(s)
        e_str = f"{e:.3f}" if isinstance(e, float) else str(e)
        f.write(f"{k},{s_str},{e_str}\n")

with open(OUT / "table2.md", "w") as f:
    f.write("# Table 2 — Simulated vs. experimental texture & microstructure metrics\n\n")
    f.write("| Metric | Simulated | Experimental |\n|---|---:|---:|\n")
    for k, s, e in rows:
        s_str = f"{s:.3f}" if isinstance(s, float) else str(s)
        e_str = f"{e:.3f}" if isinstance(e, float) else str(e)
        f.write(f"| {k} | {s_str} | {e_str} |\n")

with open(OUT / "summary.json", "w") as f:
    json.dump({
        "J_index": J_index, "M_index": M_index,
        "peak_mrd": {k: pfs[k]["peak"] for k in pfs},
        "grain_threshold_deg": GRAIN_THRESHOLD_DEG,
        "n_grains": int(len(vol_arr)),
        "median_grain_voxels": float(np.median(vol_arr)),
        "median_aspect_ratio": float(np.median(ar_arr)),
        "n_solid_voxels": int(Nvox),
        "neighbour_edges": int(len(raw_all)),
        "frac_raw_gt_5deg": float((raw_all > 5).mean()),
        "frac_diso_gt_5deg": float((diso_all > 5).mean()),
    }, f, indent=2)

print(f"[done] elapsed: {time.time() - t0:.1f}s")
print(f"       wrote: {sorted(os.listdir(OUT))}")