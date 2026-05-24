import os
import itertools
import numpy as np
import jax
import jax.numpy as jnp
from jax import lax
from jax.scipy.sparse.linalg import cg, bicgstab

# ============================================================
# JAX config: use float64 for best agreement with NumPy/SciPy
# ============================================================
jax.config.update("jax_enable_x64", True)

# ============================================================
# Domain
# ============================================================
dx = dy = dz = 0.05
nx, ny, nz = 250, 50, 50

dt = 5e-4
steps = 75001
plot_every = 10

os.makedirs("VTK", exist_ok=True)

# ============================================================
# Indexing utilities (VTK x-fastest order)
# idx = i + nx*(j + ny*k)  <=> reshape order='F'
# Fortran-order flattening in JAX via transpose.
# ============================================================
def flattenF_jax(u3: jnp.ndarray) -> jnp.ndarray:
    return jnp.ravel(jnp.transpose(u3, (2, 1, 0)))

def unflattenF_jax(u1: jnp.ndarray) -> jnp.ndarray:
    return jnp.transpose(jnp.reshape(u1, (nz, ny, nx)), (2, 1, 0))

def flatten_np(u3):
    return np.asarray(u3, dtype=float).reshape((nx, ny, nz), order="F").ravel(order="F")

def unflatten_np(u1):
    return np.asarray(u1, dtype=float).reshape((nx, ny, nz), order="F")

# ============================================================
# VTK export (STRUCTURED_POINTS)
# ============================================================
def write_vtk_structured_points(filename, fields, vectors):
    npts = nx * ny * nz
    with open(filename, "w") as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write("NumPy/SciPy phase-field output\n")
        f.write("ASCII\n")
        f.write("DATASET STRUCTURED_POINTS\n")
        f.write(f"DIMENSIONS {nx} {ny} {nz}\n")
        f.write(f"ORIGIN {dx/2.0:.8f} {dy/2.0:.8f} {dz/2.0:.8f}\n")
        f.write(f"SPACING {dx:.8f} {dy:.8f} {dz:.8f}\n")
        f.write(f"POINT_DATA {npts}\n")

        for name, arr3 in fields.items():
            f.write(f"SCALARS {name} float 1\n")
            f.write("LOOKUP_TABLE default\n")
            vals = flatten_np(arr3)
            per_line = 8
            for i in range(0, npts, per_line):
                f.write(" ".join(f"{v:.6e}" for v in vals[i:i+per_line]) + "\n")

        for name, (vx, vy, vz) in vectors.items():
            f.write(f"VECTORS {name} float\n")
            vx1, vy1, vz1 = flatten_np(vx), flatten_np(vy), flatten_np(vz)
            for i in range(npts):
                f.write(f"{vx1[i]:.6e} {vy1[i]:.6e} {vz1[i]:.6e}\n")

# ============================================================
# NEW axis-angle sampling functions (Haar-uniform SO(3))
# ============================================================
def random_axis(rng):
    v = rng.normal(size=3)
    return v / (np.linalg.norm(v) + 1e-30)

def random_angle_haar(rng):
    while True:
        w = rng.random()
        y = rng.random()
        if y <= np.sqrt(max(0.0, 1.0 - w*w)):
            return 2.0 * np.arccos(w)

def axis_angle_to_matrix(axis, theta):
    axis = np.asarray(axis, dtype=float)
    axis /= (np.linalg.norm(axis) + 1e-30)
    x, y, z = axis
    c = np.cos(theta)
    s = np.sin(theta)
    C = 1.0 - c
    R = np.array([
        [c + x*x*C,     x*y*C - z*s, x*z*C + y*s],
        [y*x*C + z*s,   c + y*y*C,   y*z*C - x*s],
        [z*x*C - y*s,   z*y*C + x*s, c + z*z*C]
    ], dtype=float)
    return R

# ============================================================
# Cubic symmetry helpers (NumPy for seeding)
# ============================================================
def misorientation_angle(R):
    tr = np.trace(R)
    c = (tr - 1.0) / 2.0
    c = np.clip(c, -1.0, 1.0)
    return np.arccos(c)

def cubic_symmetry_ops():
    ops = []
    for p in itertools.permutations(range(3)):
        for s in itertools.product([-1.0, 1.0], repeat=3):
            M = np.zeros((3, 3), dtype=float)
            for i, j in enumerate(p):
                M[i, j] = s[i]
            if np.linalg.det(M) > 0.5:
                ops.append(M)
    if len(ops) != 24:
        raise RuntimeError(f"Expected 24 cubic symmetry ops, got {len(ops)}")
    return ops

def canonical_under_cubic(R, sym_ops):
    best_R = None
    best_theta = np.inf
    for S in sym_ops:
        Ri = S @ R
        theta = misorientation_angle(Ri)
        if theta < best_theta:
            best_theta = theta
            best_R = Ri
    return best_R

def min_misorientation_under_cubic(Ra, Rb, sym_ops):
    best = np.inf
    Rt = Ra.T
    for S in sym_ops:
        ang = misorientation_angle(Rt @ (S @ Rb))
        if ang < best:
            best = ang
    return best

def generate_orientations_cubic(rng, n, min_sep_deg=0.0, max_trials=200000):
    sym_ops = cubic_symmetry_ops()
    orientations = []
    min_sep_rad = np.deg2rad(min_sep_deg)

    for _ in range(max_trials):
        axis = random_axis(rng)
        theta = random_angle_haar(rng)
        R = axis_angle_to_matrix(axis, theta)

        Rc = canonical_under_cubic(R, sym_ops)

        if min_sep_deg > 0.0 and orientations:
            ok = True
            for Rprev in orientations:
                ang = min_misorientation_under_cubic(Rprev, Rc, sym_ops)
                if ang < min_sep_rad:
                    ok = False
                    break
            if not ok:
                continue

        orientations.append(Rc)
        if len(orientations) == n:
            return orientations

    raise RuntimeError(
        f"Could not generate {n} cubic-canonical orientations with min_sep_deg={min_sep_deg} "
        f"within {max_trials} trials."
    )

# ============================================================
# Grid coordinates (cell centers) on device
# ============================================================
xs = (jnp.arange(nx) + 0.5) * dx
ys = (jnp.arange(ny) + 0.5) * dy
zs = (jnp.arange(nz) + 0.5) * dz
X3, Y3, Z3 = jnp.meshgrid(xs, ys, zs, indexing="ij")

# ============================================================
# FiPy-like arithmeticFaceValue (JAX)
# ============================================================
def arithmetic_face_value_jax(u, axis: int):
    if axis == 0:
        avg = 0.5 * (u[0:nx-1] + u[1:nx])
        return jnp.concatenate([u[0:1], avg, u[nx-1:nx]], axis=0)
    if axis == 1:
        avg = 0.5 * (u[:, 0:ny-1] + u[:, 1:ny])
        return jnp.concatenate([u[:, 0:1], avg, u[:, ny-1:ny]], axis=1)
    if axis == 2:
        avg = 0.5 * (u[:, :, 0:nz-1] + u[:, :, 1:nz])
        return jnp.concatenate([u[:, :, 0:1], avg, u[:, :, nz-1:nz]], axis=2)
    raise ValueError("axis must be 0,1,2")

def cell_grad_gauss_jax(u):
    ufx = arithmetic_face_value_jax(u, 0)
    ufy = arithmetic_face_value_jax(u, 1)
    ufz = arithmetic_face_value_jax(u, 2)
    gx = (ufx[1:] - ufx[:-1]) / dx
    gy = (ufy[:, 1:] - ufy[:, :-1]) / dy
    gz = (ufz[:, :, 1:] - ufz[:, :, :-1]) / dz
    return gx, gy, gz

def face_grad_fipy_jax(u):
    gxC, gyC, gzC = cell_grad_gauss_jax(u)

    def avg_to_xfaces(a):
        avg = 0.5 * (a[0:nx-1] + a[1:nx])
        return jnp.concatenate([a[0:1], avg, a[nx-1:nx]], axis=0)

    def avg_to_yfaces(a):
        avg = 0.5 * (a[:, 0:ny-1] + a[:, 1:ny])
        return jnp.concatenate([a[:, 0:1], avg, a[:, ny-1:ny]], axis=1)

    def avg_to_zfaces(a):
        avg = 0.5 * (a[:, :, 0:nz-1] + a[:, :, 1:nz])
        return jnp.concatenate([a[:, :, 0:1], avg, a[:, :, nz-1:nz]], axis=2)

    gx_x = avg_to_xfaces(gxC); gy_x = avg_to_xfaces(gyC); gz_x = avg_to_xfaces(gzC)
    gx_y = avg_to_yfaces(gxC); gy_y = avg_to_yfaces(gyC); gz_y = avg_to_yfaces(gzC)
    gx_z = avg_to_zfaces(gxC); gy_z = avg_to_zfaces(gyC); gz_z = avg_to_zfaces(gzC)

    gn_x = jnp.zeros((nx+1, ny, nz), dtype=u.dtype)
    gn_x = gn_x.at[1:nx].set((u[1:nx] - u[0:nx-1]) / dx)

    gn_y = jnp.zeros((nx, ny+1, nz), dtype=u.dtype)
    gn_y = gn_y.at[:, 1:ny].set((u[:, 1:ny] - u[:, 0:ny-1]) / dy)

    gn_z = jnp.zeros((nx, ny, nz+1), dtype=u.dtype)
    gn_z = gn_z.at[:, :, 1:nz].set((u[:, :, 1:nz] - u[:, :, 0:nz-1]) / dz)

    gFx = jnp.stack([gn_x, gy_x, gz_x], axis=-1)
    gFy = jnp.stack([gx_y, gn_y, gz_y], axis=-1)
    gFz = jnp.stack([gx_z, gy_z, gn_z], axis=-1)
    return gFx, gFy, gFz

def face_mag_jax(F):
    return jnp.sqrt(F[..., 0]**2 + F[..., 1]**2 + F[..., 2]**2)

# ============================================================
# Matrix-free diffusion operator
# ============================================================
def apply_A_diff_no_bc(Dxf, Dyf, Dzf, u):
    """
    Natural / zero-flux handling at outer boundaries:
    only internal faces contribute to flux divergence.
    """
    res = jnp.zeros_like(u)

    kx = Dxf[1:nx] / (dx * dx)
    du = u[1:nx] - u[0:nx-1]
    res = res.at[0:nx-1].add(-kx * du)
    res = res.at[1:nx].add( kx * du)

    ky = Dyf[:, 1:ny] / (dy * dy)
    du = u[:, 1:ny] - u[:, 0:ny-1]
    res = res.at[:, 0:ny-1].add(-ky * du)
    res = res.at[:, 1:ny].add( ky * du)

    kz = Dzf[:, :, 1:nz] / (dz * dz)
    du = u[:, :, 1:nz] - u[:, :, 0:nz-1]
    res = res.at[:, :, 0:nz-1].add(-kz * du)
    res = res.at[:, :, 1:nz].add( kz * du)

    return res

# ============================================================
# Dirichlet contribution helpers (cell-centered FV)
# We will TURN THESE ON only when (t > 0.505).
# Faces: x=0, y=0, z=0, z=Lz (i.e. face k=nz)
# ============================================================
def apply_A_diff_dirichlet_faces_scaled(Dxf, Dyf, Dzf, u, bc_on_f):
    """
    Returns A(u) for diffusion with Dirichlet elimination on:
      x=0, y=0, z=0, z=Lz,
    scaled by bc_on_f in [0,1]. If bc_on_f=0 => same as no_bc.
    NOTE: Dirichlet value itself goes into RHS, not into this operator.
    """
    res = apply_A_diff_no_bc(Dxf, Dyf, Dzf, u)

    # extra diagonal-like terms from boundary elimination:
    kb_x0 = (2.0 * Dxf[0, :, :]  / (dx * dx))   # acts on u[0,:,:]
    kb_y0 = (2.0 * Dyf[:, 0, :]  / (dy * dy))   # acts on u[:,0,:]
    kb_z0 = (2.0 * Dzf[:, :, 0]  / (dz * dz))   # acts on u[:,:,0]
    kb_z1 = (2.0 * Dzf[:, :, nz] / (dz * dz))   # acts on u[:,:,nz-1]

    res = res.at[0].add(      bc_on_f * kb_x0 * u[0])
    res = res.at[:, 0].add(   bc_on_f * kb_y0 * u[:, 0])
    res = res.at[:, :, 0].add(bc_on_f * kb_z0 * u[:, :, 0])
    res = res.at[:, :, nz-1].add(bc_on_f * kb_z1 * u[:, :, nz-1])

    return res

def rhs_dirichlet_faces_3d(Dxf, Dyf, Dzf, bc_x0, bc_y0, bc_z0, bc_z1):
    """
    RHS contribution from Dirichlet elimination on x=0,y=0,z=0,z=Lz.
    """
    rhs = jnp.zeros((nx, ny, nz), dtype=jnp.float64)

    kb_x0 = (2.0 * Dxf[0, :, :]  / (dx * dx))
    kb_y0 = (2.0 * Dyf[:, 0, :]  / (dy * dy))
    kb_z0 = (2.0 * Dzf[:, :, 0]  / (dz * dz))
    kb_z1 = (2.0 * Dzf[:, :, nz] / (dz * dz))

    rhs = rhs.at[0].add(kb_x0 * bc_x0)
    rhs = rhs.at[:, 0].add(kb_y0 * bc_y0)
    rhs = rhs.at[:, :, 0].add(kb_z0 * bc_z0)
    rhs = rhs.at[:, :, nz-1].add(kb_z1 * bc_z1)

    return rhs

def diag_diff_no_bc(Dxf, Dyf, Dzf):
    diag = jnp.zeros((nx, ny, nz), dtype=jnp.float64)

    kx = Dxf[1:nx] / (dx * dx)
    diag = diag.at[0:nx-1].add(kx)
    diag = diag.at[1:nx].add(kx)

    ky = Dyf[:, 1:ny] / (dy * dy)
    diag = diag.at[:, 0:ny-1].add(ky)
    diag = diag.at[:, 1:ny].add(ky)

    kz = Dzf[:, :, 1:nz] / (dz * dz)
    diag = diag.at[:, :, 0:nz-1].add(kz)
    diag = diag.at[:, :, 1:nz].add(kz)

    return diag

def diag_diff_dirichlet_faces(Dxf, Dyf, Dzf):
    """
    Diagonal for diffusion with Dirichlet elimination on x=0,y=0,z=0,z=Lz.
    """
    diag = diag_diff_no_bc(Dxf, Dyf, Dzf)
    diag = diag.at[0].add(      2.0 * Dxf[0, :, :]  / (dx * dx))
    diag = diag.at[:, 0].add(   2.0 * Dyf[:, 0, :]  / (dy * dy))
    diag = diag.at[:, :, 0].add(2.0 * Dzf[:, :, 0]  / (dz * dz))
    diag = diag.at[:, :, nz-1].add(2.0 * Dzf[:, :, nz] / (dz * dz))
    return diag

# ============================================================
# Physics parameters
# ============================================================
laser_power = 2000.0
laser_speed = 1.0

# -------------------------------------------------------
# 3-track laser parameters
#   Track 1 (laser_step    0..24999): z_center = nz*dz/4
#   Track 2 (laser_step 25000..49999): z_center = nz*dz/2
#   Track 3 (laser_step 50000+      ): z_center = 3*nz*dz/4
# All tracks share: y_center = ny*dy, x_center0 = nx*dx/8
# x_center resets to x_center0 at the start of each track.
# -------------------------------------------------------
track_steps = 25000          # steps per laser track
y_center_all = ny * dy       # same for all tracks
x_center0_all = nx * dx / 8  # same starting x for all tracks

# z_center values for each track (used inside one_step via jnp.where)
z_center_track1 = nz * dz / 4.0
z_center_track2 = nz * dz / 2.0
z_center_track3 = 3.0 * nz * dz / 4.0

DT = 2.48
q0 = 0.0
T_0 = -0.1
source_coeff0 = 0.0

def gaussian_source_jax(x, y, z, x_center, y_center, z_center, power):
    a = 0.02
    b = 0.02
    c = 0.02
    absorption = 0.3
    norm = absorption * power / ((2.0 * jnp.pi) ** 1.5 * a * b * c)
    return norm * jnp.exp(
        -((x - x_center) ** 2) / (2.0 * a ** 2)
        -((y - y_center) ** 2) / (2.0 * b ** 2)
        -((z - z_center) ** 2) / (2.0 * c ** 2)
    )

alpha = 0.015
tau_phase = 3e-4
kappa1 = 0.9
kappa2 = 20.0
epsilon = 0.008
s_gb = 0.01

tiny = 1e-20
eps_a = 0.05
a_min = 0.005
a_max = 10.0

tau_P = 3e-3
mu = 1e3
beta_theta = 1e5
T_mod0 = 0.0

thetaSmallValue = 1e-6
eps_floor = 1e-12

# ============================================================
# Liquid (phase==0) orientation is "0" => Identity rotation
# ============================================================
phi_liquid = 1e-3
ID_P9 = jnp.array([1.,0.,0., 0.,1.,0., 0.,0.,1.], dtype=jnp.float64)[:, None, None, None]

# ============================================================
# Anisotropy coefficients
# ============================================================
def update_anisotropy_coeffs_jax(phase_old, P9_old):
    gFx, gFy, gFz = face_grad_fipy_jax(phase_old)
    mx = face_mag_jax(gFx) + tiny
    my = face_mag_jax(gFy) + tiny
    mz = face_mag_jax(gFz) + tiny

    nxF, nyF, nzF = gFx[..., 0]/mx, gFx[..., 1]/mx, gFx[..., 2]/mx
    nxG, nyG, nzG = gFy[..., 0]/my, gFy[..., 1]/my, gFy[..., 2]/my
    nxH, nyH, nzH = gFz[..., 0]/mz, gFz[..., 1]/mz, gFz[..., 2]/mz

    Pxf = jax.vmap(lambda p: arithmetic_face_value_jax(p, 0), in_axes=0)(P9_old)
    Pyf = jax.vmap(lambda p: arithmetic_face_value_jax(p, 1), in_axes=0)(P9_old)
    Pzf = jax.vmap(lambda p: arithmetic_face_value_jax(p, 2), in_axes=0)(P9_old)

    def a_faces(Pf, nx_, ny_, nz_):
        nc1 = Pf[0]*nx_ + Pf[3]*ny_ + Pf[6]*nz_
        nc2 = Pf[1]*nx_ + Pf[4]*ny_ + Pf[7]*nz_
        nc3 = Pf[2]*nx_ + Pf[5]*ny_ + Pf[8]*nz_
        ncm = jnp.sqrt(nc1*nc1 + nc2*nc2 + nc3*nc3) + 1e-12
        nc1n = nc1/ncm; nc2n = nc2/ncm; nc3n = nc3/ncm
        S4 = nc1n**4 + nc2n**4 + nc3n**4
        a_expr = (1 - 3*eps_a) * (1.0 + (4*eps_a*S4)/(1 - 3*eps_a))
        return jnp.clip(a_expr, a_min, a_max)

    ax = a_faces(Pxf, nxF, nyF, nzF)
    ay = a_faces(Pyf, nxG, nyG, nzG)
    az = a_faces(Pzf, nxH, nyH, nzH)

    Dph_xf = (alpha**2) * (ax**2)
    Dph_yf = (alpha**2) * (ay**2)
    Dph_zf = (alpha**2) * (az**2)
    return Dph_xf, Dph_yf, Dph_zf

# ============================================================
# SO(3) projection (batched SVD)
# ============================================================
def project_P9_to_SO3_jax(P9, tol=1e-3):
    Pmat = jnp.stack([
        jnp.stack([P9[0], P9[1], P9[2]], axis=-1),
        jnp.stack([P9[3], P9[4], P9[5]], axis=-1),
        jnp.stack([P9[6], P9[7], P9[8]], axis=-1),
    ], axis=-2)

    Pmat_T = jnp.transpose(Pmat, (2, 1, 0, 3, 4))
    M = jnp.reshape(Pmat_T, (nx*ny*nz, 3, 3))

    U, _, Vh = jnp.linalg.svd(M, full_matrices=False)
    R = jnp.matmul(U, Vh)
    detR = jnp.linalg.det(R)
    bad = detR < 0.0

    sign = jnp.where(bad, -1.0, 1.0)
    U2 = U.at[:, :, 2].set(U[:, :, 2] * sign[:, None])
    R2 = jnp.matmul(U2, Vh)

    R2 = jnp.where(jnp.abs(R2) < tol, 0.0, R2)

    R2_T = jnp.reshape(R2, (nz, ny, nx, 3, 3))
    Rmat = jnp.transpose(R2_T, (2, 1, 0, 3, 4))

    P9_out = jnp.stack([
        Rmat[..., 0, 0], Rmat[..., 0, 1], Rmat[..., 0, 2],
        Rmat[..., 1, 0], Rmat[..., 1, 1], Rmat[..., 1, 2],
        Rmat[..., 2, 0], Rmat[..., 2, 1], Rmat[..., 2, 2],
    ], axis=0)
    return P9_out

# ============================================================
# Krylov solvers
# ============================================================
def solve_linear_cg(A_mv, b, x0, M_diag, tol=1e-12, maxiter=500):
    def M_mv(v):
        return v / M_diag
    x, info = cg(A_mv, b, x0=x0, tol=tol, atol=0.0, maxiter=maxiter, M=M_mv)
    return x, info

def solve_linear_bicgstab(A_mv, b, x0, M_diag, tol=1e-12, maxiter=800):
    def M_mv(v):
        return v / M_diag
    x, info = bicgstab(A_mv, b, x0=x0, tol=tol, atol=0.0, maxiter=maxiter, M=M_mv)
    return x, info

# ============================================================
# dT diffusion coefficients (constants)
# ============================================================
DT_xf = DT * jnp.ones((nx+1, ny, nz), dtype=jnp.float64)
DT_yf = DT * jnp.ones((nx, ny+1, nz), dtype=jnp.float64)
DT_zf = DT * jnp.ones((nx, ny, nz+1), dtype=jnp.float64)

# Precompute BOTH diagonals and RHS for BC, then blend in time:
diag_dT_diff_no  = diag_diff_no_bc(DT_xf, DT_yf, DT_zf)
diag_dT_diff_dir = diag_diff_dirichlet_faces(DT_xf, DT_yf, DT_zf)

bc_T_val = jnp.array(-3.0, dtype=jnp.float64)
rhs_dT_bc_dir = rhs_dirichlet_faces_3d(DT_xf, DT_yf, DT_zf, bc_T_val, bc_T_val, bc_T_val, bc_T_val)

# ============================================================
#  ### >>> CUBIC SYMMETRY CHANGE (device constants + helpers)
# ============================================================
_CUBIC_OPS_NP = np.stack(cubic_symmetry_ops(), axis=0).astype(np.float64)
CUBIC_OPS = jnp.asarray(_CUBIC_OPS_NP, dtype=jnp.float64)  # (24,3,3)

def P9_to_R_jax(P9):
    return jnp.stack([
        jnp.stack([P9[0], P9[1], P9[2]], axis=-1),
        jnp.stack([P9[3], P9[4], P9[5]], axis=-1),
        jnp.stack([P9[6], P9[7], P9[8]], axis=-1),
    ], axis=-2)

def R_to_P9_jax(R):
    return jnp.stack([
        R[..., 0, 0], R[..., 0, 1], R[..., 0, 2],
        R[..., 1, 0], R[..., 1, 1], R[..., 1, 2],
        R[..., 2, 0], R[..., 2, 1], R[..., 2, 2],
    ], axis=0)

def _angle_from_relR(Rrel):
    tr = Rrel[..., 0, 0] + Rrel[..., 1, 1] + Rrel[..., 2, 2]
    c = (tr - 1.0) / 2.0
    c = jnp.clip(c, -1.0, 1.0)
    return jnp.arccos(c)

def cubic_min_misorientation_pair_jax(Ra, Rb, ops=CUBIC_OPS):
    RaT = jnp.swapaxes(Ra, -1, -2)
    best0 = jnp.full(Ra.shape[:-2], jnp.inf, dtype=jnp.float64)

    def body(i, best):
        S = ops[i]
        SRb = jnp.einsum("ij,...jk->...ik", S, Rb)
        Rrel = jnp.einsum("...ij,...jk->...ik", RaT, SRb)
        ang = _angle_from_relR(Rrel)
        return jnp.minimum(best, ang)

    return lax.fori_loop(0, ops.shape[0], body, best0)

def cubic_misorientation_faces_jax(P9, ops=CUBIC_OPS):
    R = P9_to_R_jax(P9)

    Pg_x = jnp.zeros((nx+1, ny, nz), dtype=jnp.float64)
    Pg_y = jnp.zeros((nx, ny+1, nz), dtype=jnp.float64)
    Pg_z = jnp.zeros((nx, ny, nz+1), dtype=jnp.float64)

    Ra = R[0:nx-1, :, :, :, :]
    Rb = R[1:nx,   :, :, :, :]
    ang = cubic_min_misorientation_pair_jax(Ra, Rb, ops=ops) / dx
    Pg_x = Pg_x.at[1:nx].set(ang)

    Ra = R[:, 0:ny-1, :, :, :]
    Rb = R[:, 1:ny,   :, :, :]
    ang = cubic_min_misorientation_pair_jax(Ra, Rb, ops=ops) / dy
    Pg_y = Pg_y.at[:, 1:ny].set(ang)

    Ra = R[:, :, 0:nz-1, :, :]
    Rb = R[:, :, 1:nz,   :, :]
    ang = cubic_min_misorientation_pair_jax(Ra, Rb, ops=ops) / dz
    Pg_z = Pg_z.at[:, :, 1:nz].set(ang)

    return Pg_x, Pg_y, Pg_z

def cubic_gradmag_cell_from_faces(Pg_x, Pg_y, Pg_z):
    gx = 0.5 * (Pg_x[0:nx] + Pg_x[1:nx+1])
    gy = 0.5 * (Pg_y[:, 0:ny] + Pg_y[:, 1:ny+1])
    gz = 0.5 * (Pg_z[:, :, 0:nz] + Pg_z[:, :, 1:nz+1])
    return jnp.sqrt(gx*gx + gy*gy + gz*gz)

def cubic_gauge_fix_jax(P9_ref, P9_new, ops=CUBIC_OPS):
    Rref = P9_to_R_jax(P9_ref)
    Rnew = P9_to_R_jax(P9_new)

    best_score0 = jnp.full(Rref.shape[:-2], -jnp.inf, dtype=jnp.float64)
    best_R0 = Rnew

    def body(i, carry):
        best_score, best_R = carry
        S = ops[i]
        cand = jnp.einsum("ij,...jk->...ik", S, Rnew)
        score = jnp.sum(Rref * cand, axis=(-2, -1))
        better = score > best_score
        best_score = jnp.where(better, score, best_score)
        best_R = jnp.where(better[..., None, None], cand, best_R)
        return (best_score, best_R)

    _, Rbest = lax.fori_loop(0, ops.shape[0], body, (best_score0, best_R0))
    return R_to_P9_jax(Rbest)

# ============================================================
#  ### >>> CUBIC SYMMETRY CHANGE END
# ============================================================

@jax.jit
def one_step(phase, dT, P9, Pfunc, t, laser_step):
    cond1 = (t > 0.0) & (t < 0.500)
    cond2 = (t > 0.500) & (t <= 0.505)
    cond3 = (t > 0.505)   # <-- after this, TURN ON Dirichlet on the 4 faces

    q = jnp.where(cond1, 100.0, 0)
    T_mod = jnp.where(cond3, 1.0, 0.0)
    source_coeff = jnp.where(cond3, 1.0, 0.0)

    # your existing "global reset" during 0.500<t<=0.505
    dT = jnp.where(cond2, -3.0, dT)

    # ----------------------------------------------------------
    # 3-track laser: determine current track from laser_step
    #   Track 1: laser_step in [0, track_steps)
    #   Track 2: laser_step in [track_steps, 2*track_steps)
    #   Track 3: laser_step in [2*track_steps, ...)
    # ----------------------------------------------------------
    track2_on = (laser_step >= track_steps)
    track3_on = (laser_step >= 2 * track_steps)

    # z_center depends on current track
    z_center_cur = jnp.where(track3_on, z_center_track3,
                     jnp.where(track2_on, z_center_track2,
                               z_center_track1))

    # Steps elapsed within the current track (for x_center computation)
    steps_in_track = jnp.where(track3_on, laser_step - 2 * track_steps,
                       jnp.where(track2_on, laser_step - track_steps,
                                 laser_step))

    # x_center resets to x_center0 at the start of each track
    x_center = x_center0_all + steps_in_track * laser_speed * dt

    # Increment laser_step when the laser is active (cond3)
    laser_step = laser_step + jnp.where(cond3, 1, 0).astype(jnp.int64)

    source_term = lax.cond(
        x_center <= (nx * dx),
        lambda _: gaussian_source_jax(X3, Y3, Z3, x_center, y_center_all, z_center_cur, laser_power),
        lambda _: jnp.zeros((nx, ny, nz), dtype=jnp.float64),
        operand=None
    )

    phase_old = phase
    dT_old = dT
    P9_old = P9

    Dph_xf, Dph_yf, Dph_zf = update_anisotropy_coeffs_jax(phase_old, P9_old)
    diag_ph_diff = diag_diff_no_bc(Dph_xf, Dph_yf, Dph_zf)

    Pg_x_faces, Pg_y_faces, Pg_z_faces = cubic_misorientation_faces_jax(P9_old, ops=CUBIC_OPS)
    Pgrad_cell_now = cubic_gradmag_cell_from_faces(Pg_x_faces, Pg_y_faces, Pg_z_faces)

    coeffF = ((phase - 0.5 - (kappa1/jnp.pi)*jnp.arctan(kappa2*dT))*(1.0 - phase)
              - (2.0*s_gb + (epsilon**2)*Pgrad_cell_now) * Pgrad_cell_now)

    diag_phi = (tau_phase/dt) - coeffF
    diag_phi_1d = flattenF_jax(diag_ph_diff + diag_phi)

    def A_phi_mv(x1d):
        u = unflattenF_jax(x1d)
        Au = apply_A_diff_no_bc(Dph_xf, Dph_yf, Dph_zf, u) + diag_phi * u
        return flattenF_jax(Au)

    b_phi = (tau_phase/dt) * flattenF_jax(phase_old)
    x0_phi = flattenF_jax(phase)

    phi_new_1d, _ = solve_linear_bicgstab(A_phi_mv, b_phi, x0_phi, diag_phi_1d, tol=1e-12, maxiter=800)
    phase = unflattenF_jax(phi_new_1d)

    # ============================================================
    # dT solve with Dirichlet ON ONLY WHEN (t > 0.505)
    # Dirichlet value: T=-3 at x=0, y=0, z=0, z=Lz
    # ============================================================
    bc_on_f = jnp.where(cond3, 1.0, 0.0).astype(jnp.float64)

    # Blend the diagonal and RHS contributions in time (no branches in linear algebra):
    diag_dT_diff = diag_dT_diff_no + bc_on_f * (diag_dT_diff_dir - diag_dT_diff_no)
    rhs_dT_bc = bc_on_f * rhs_dT_bc_dir

    diag_T = diag_dT_diff + ((1.0/dt) + q)
    diag_T_1d = flattenF_jax(diag_T)

    def A_T_mv(x1d):
        u = unflattenF_jax(x1d)
        Au = apply_A_diff_dirichlet_faces_scaled(DT_xf, DT_yf, DT_zf, u, bc_on_f) + ((1.0/dt) + q) * u
        return flattenF_jax(Au)

    b_T = (1.0/dt) * flattenF_jax(dT_old) \
          - (1.0/dt) * flattenF_jax(phase_old) \
          + q * T_0 \
          + source_coeff * flattenF_jax(source_term) \
          + flattenF_jax(rhs_dT_bc) \
          + (1.0/dt) * flattenF_jax(phase)

    x0_T = flattenF_jax(dT)
    dT_new_1d, _ = solve_linear_cg(A_T_mv, b_T, x0_T, diag_T_1d, tol=1e-12, maxiter=500)
    dT = unflattenF_jax(dT_new_1d)

    expo_val = epsilon * beta_theta * Pgrad_cell_now
    expo_val = jnp.minimum(expo_val, 100.0)
    Pfunc = 1.0 + jnp.exp(-expo_val) * (mu / epsilon - 1.0)

    phaseMod = phase + (phase < thetaSmallValue) * thetaSmallValue
    mobility = tau_P * (phaseMod**2) * Pfunc * jnp.exp(-T_mod * dT)
    mass_diag = mobility / dt

    phx2 = arithmetic_face_value_jax(phase, 0)**2
    phy2 = arithmetic_face_value_jax(phase, 1)**2
    phz2 = arithmetic_face_value_jax(phase, 2)**2

    Pg_x = Pg_x_faces + (Pg_x_faces < eps_floor) * eps_floor
    Pg_y = Pg_y_faces + (Pg_y_faces < eps_floor) * eps_floor
    Pg_z = Pg_z_faces + (Pg_z_faces < eps_floor) * eps_floor

    Kx = s_gb * (phx2 / Pg_x) + (epsilon**2) * phx2
    Ky = s_gb * (phy2 / Pg_y) + (epsilon**2) * phy2
    Kz = s_gb * (phz2 / Pg_z) + (epsilon**2) * phz2

    diagK = diag_diff_no_bc(Kx, Ky, Kz)
    diagA_common = diagK + mass_diag
    diagA_common_1d = flattenF_jax(diagA_common)

    def orient_body(k, P9_curr):
        def A_mv(x1d):
            u = unflattenF_jax(x1d)
            Au = apply_A_diff_no_bc(Kx, Ky, Kz, u) + mass_diag * u
            return flattenF_jax(Au)

        b = flattenF_jax(mass_diag * P9_old[k])
        x0 = flattenF_jax(P9_curr[k])

        sol_1d, _ = solve_linear_cg(A_mv, b, x0, diagA_common_1d, tol=1e-12, maxiter=500)
        sol = unflattenF_jax(sol_1d)
        return P9_curr.at[k].set(sol)

    P9 = lax.fori_loop(0, 9, orient_body, P9)

    P9 = project_P9_to_SO3_jax(P9, tol=1e-3)
    P9 = cubic_gauge_fix_jax(P9_old, P9, ops=CUBIC_OPS)

    liquid_mask = phase < phi_liquid
    P9 = jnp.where(liquid_mask[None, ...], ID_P9, P9)

    t = t + dt
    return phase, dT, P9, Pfunc, t, laser_step

# ============================================================
# Fields
# ============================================================
phase_np = np.zeros((nx, ny, nz), dtype=float)
dT_np    = np.full((nx, ny, nz), -0.5, dtype=float)

P11 = np.ones((nx, ny, nz), dtype=float); P12 = np.zeros((nx, ny, nz), dtype=float); P13 = np.zeros((nx, ny, nz), dtype=float)
P21 = np.zeros((nx, ny, nz), dtype=float); P22 = np.ones((nx, ny, nz), dtype=float); P23 = np.zeros((nx, ny, nz), dtype=float)
P31 = np.zeros((nx, ny, nz), dtype=float); P32 = np.zeros((nx, ny, nz), dtype=float); P33 = np.ones((nx, ny, nz), dtype=float)

Pfunc_np = np.ones((nx, ny, nz), dtype=float)

rng = np.random.default_rng(12345)
numSeeds = 180
Rseed = 5.0 * dx
rx = ry = rz = 5.0 * dx

xmin, xmax = Rseed, nx*dx - Rseed
ymin, ymax = Rseed, ny*dy - Rseed
zmin, zmax = Rseed, nz*dz - Rseed

xs_np = (np.arange(nx) + 0.5) * dx
ys_np = (np.arange(ny) + 0.5) * dy
zs_np = (np.arange(nz) + 0.5) * dz
X3_np, Y3_np, Z3_np = np.meshgrid(xs_np, ys_np, zs_np, indexing="ij")

# ============================================================
# Seed initialization WITH cubic symmetry considerations (unchanged)
# ============================================================
orientations = generate_orientations_cubic(rng, n=numSeeds, min_sep_deg=5.0)

max_place_tries = 20000  # per seed
placed = 0

for i in range(numSeeds):
    Rmat = orientations[i]

    for _try in range(max_place_tries):
        cx = rng.uniform(xmin, xmax)
        cy = rng.uniform(ymin, ymax)
        cz = rng.uniform(zmin, zmax)

        X = X3_np - cx
        Y = Y3_np - cy
        Z = Z3_np - cz

        xl = Rmat[0,0]*X + Rmat[1,0]*Y + Rmat[2,0]*Z
        yl = Rmat[0,1]*X + Rmat[1,1]*Y + Rmat[2,1]*Z
        zl = Rmat[0,2]*X + Rmat[1,2]*Y + Rmat[2,2]*Z

        seed = (xl/rx)**2 + (yl/ry)**2 + (zl/rz)**2 < 1.0
        empty = phase_np < 0.5
        seed_new = seed & empty

        if not np.any(seed_new):
            continue

        phase_np[seed_new] = 1.0

        P11[seed_new], P12[seed_new], P13[seed_new] = Rmat[0,0], Rmat[0,1], Rmat[0,2]
        P21[seed_new], P22[seed_new], P23[seed_new] = Rmat[1,0], Rmat[1,1], Rmat[1,2]
        P31[seed_new], P32[seed_new], P33[seed_new] = Rmat[2,0], Rmat[2,1], Rmat[2,2]

        placed += 1
        break
    else:
        print(f"WARNING: Could not place seed {i} after {max_place_tries} tries.")

print(f"Placed {placed}/{numSeeds} seeds.")

# ============================================================
# Move to device
# ============================================================
phase = jnp.asarray(phase_np, dtype=jnp.float64)
dT    = jnp.asarray(dT_np, dtype=jnp.float64)
Pfunc = jnp.asarray(Pfunc_np, dtype=jnp.float64)

P9 = jnp.stack([
    jnp.asarray(P11), jnp.asarray(P12), jnp.asarray(P13),
    jnp.asarray(P21), jnp.asarray(P22), jnp.asarray(P23),
    jnp.asarray(P31), jnp.asarray(P32), jnp.asarray(P33),
], axis=0).astype(jnp.float64)

# ============================================================
# Main time loop (Python for VTK I/O; compute per-step on GPU)
# ============================================================
t = jnp.array(0.0, dtype=jnp.float64)
laser_step = jnp.array(0, dtype=jnp.int64)

for step in range(1, steps + 1):
    phase, dT, P9, Pfunc, t, laser_step = one_step(phase, dT, P9, Pfunc, t, laser_step)

    if (step % plot_every) == 0 or step == steps:
        phase_h = np.array(jax.device_get(phase))
        dT_h = np.array(jax.device_get(dT))
        P9_h = np.array(jax.device_get(P9))

        P11h, P12h, P13h = P9_h[0], P9_h[1], P9_h[2]
        P21h, P22h, P23h = P9_h[3], P9_h[4], P9_h[5]
        P31h, P32h, P33h = P9_h[6], P9_h[7], P9_h[8]

        tr = P11h + P22h + P33h
        cang = np.clip((tr - 1.0) / 2.0, -1.0, 1.0)
        angn_over_pi = np.arccos(cang) / np.pi

        vtk_filename = os.path.join("VTK", f"fields_{step:06d}.vtk")

        if 0.500 < float(t) < 0.510 or step == 25000 or step == 50000 or step == 75000:
            fields = {
                "phi": phase_h,
                "dT": dT_h,
                "angn_over_pi": angn_over_pi,
                "P11h": P11h,
                "P12h": P12h,
                "P13h": P13h,
                "P21h": P21h,
                "P22h": P22h,
                "P23h": P23h,
                "P31h": P31h,
                "P32h": P32h,
                "P33h": P33h,
            }
        else:
            fields = {
                "phi": phase_h,
                "dT": dT_h,
                "angn_over_pi": angn_over_pi,
            }

        vectors = {}
        write_vtk_structured_points(vtk_filename, fields, vectors)

        # Print track info for monitoring
        ls_val = int(jax.device_get(laser_step))
        if ls_val < track_steps:
            track_label = "Track 1 (z=nz*dz/4)"
        elif ls_val < 2 * track_steps:
            track_label = "Track 2 (z=nz*dz/2)"
        else:
            track_label = "Track 3 (z=3*nz*dz/4)"
        print(f"Saved VTK at t = {float(t):.6f}, laser_step = {ls_val}, {track_label} -> {vtk_filename}")

print("Done.")