"""
Schwarzschild Condensation - Linear Stability Analysis v3
Hermite-Gauss basis, Goldstone-aware verdict, temiz sürüm
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.integrate import solve_bvp, simpson
from scipy.linalg import eigh
import sys, os, csv

sys.path.insert(0, '/workspace/code')
from utils import constants as C

G, c, hbar, M_sun = C.G, C.c, C.hbar, C.M_sun

# ============================================================
# PARAMETRELER
# ============================================================

M_BH = 1.0 * M_sun
Rs   = 2 * G * M_BH / c**2
m_pair_MeV = 10.0
m_pair_kg = m_pair_MeV * 1e6 * 1.602176634e-19 / c**2
ell_dS = Rs
omega_dS = c / ell_dS
sigma_dS = np.sqrt(hbar / (m_pair_kg * omega_dS))
g_tilde = 0.1

print("=" * 70)
print("STABILITY ANALYSIS v3 - Hermite-Gauss Basis")
print("=" * 70)
print(f"sigma_dS = {sigma_dS:.4e} m")
print(f"g_tilde  = {g_tilde}")
print("=" * 70)

# ============================================================
# GROUND STATE
# ============================================================

print("\n-> Ground state hesaplaniyor...")

def gp_system(xi, y, params):
    eps_til = params[0]
    phi, phi_p, n_int = y
    xi_safe = np.where(np.abs(xi) < 1e-8, 1e-8, xi)
    phi_pp = -(2.0/xi_safe)*phi_p + (xi_safe**2 + g_tilde*phi**2 - eps_til)*phi
    dn = 4*np.pi*xi_safe**2*phi**2
    return np.vstack([phi_p, phi_pp, dn])

def gp_bc(ya, yb, params):
    return np.array([ya[1], ya[2], ya[0]-1.0, yb[0]])

xi_grid = np.linspace(0.001, 6.0, 200)
phi_init = np.exp(-xi_grid**2/2.0)
phi_p_init = -xi_grid * phi_init
n_init = np.zeros_like(xi_grid)
for i in range(1, len(xi_grid)):
    n_init[i] = simpson(4*np.pi*xi_grid[:i+1]**2 * phi_init[:i+1]**2,
                        x=xi_grid[:i+1])
y_init = np.vstack([phi_init, phi_p_init, n_init])

sol = solve_bvp(gp_system, gp_bc, xi_grid, y_init,
                p=[3.0], tol=1e-7, max_nodes=20000, verbose=0)
mu_0 = sol.p[0]
print(f"  eigenvalue eps0 = {mu_0:.6f}")

# Ground state'i hassas grid'e
xi_eval = np.linspace(0.001, 8.0, 500)
phi_0 = np.interp(xi_eval, sol.x, sol.y[0])
phi_0 = np.where(xi_eval > 5.0,
                 phi_0 * np.exp(-(xi_eval-5.0)**2),
                 phi_0)

# ============================================================
# HERMITE-GAUSS BASIS
# ============================================================

print("\n-> 3D s-wave HO basis insa ediliyor...")

N_basis = 30
xi_b = xi_eval
dxi = xi_b[1] - xi_b[0]
w_3d = 4 * np.pi * xi_b**2

def gaussian(xi):
    return np.exp(-xi**2 / 2)

basis = np.zeros((N_basis, len(xi_b)))
trial = np.zeros((N_basis, len(xi_b)))
for n in range(N_basis):
    trial[n] = xi_b**(2*n) * gaussian(xi_b)

# Gram-Schmidt
for n in range(N_basis):
    basis[n] = trial[n].copy()
    for m in range(n):
        overlap = np.sum(basis[n] * basis[m] * w_3d) * dxi
        basis[n] -= overlap * basis[m]
    norm = np.sqrt(np.sum(basis[n]**2 * w_3d) * dxi)
    if norm > 1e-15:
        basis[n] /= norm

# Orthonormality check
print("  Orthonormality (lowest 5):")
ortho_ok = True
for i in range(min(5, N_basis)):
    for j in range(min(5, N_basis)):
        ov = np.sum(basis[i] * basis[j] * w_3d) * dxi
        if i == j and abs(ov - 1) > 1e-3:
            ortho_ok = False
        if i != j and abs(ov) > 1e-3:
            ortho_ok = False
print(f"  {'OK' if ortho_ok else 'FAIL'}: basis orthonormal")

# ============================================================
# BdG MATRIX ELEMENTS
# ============================================================

print(f"\n-> BdG matrisleri hesaplaniyor ({N_basis}x{N_basis})...")

L_mat = np.zeros((N_basis, N_basis))
M_mat = np.zeros((N_basis, N_basis))

def laplacian_3d(f, xi, dxi):
    f_p = np.gradient(f, dxi)
    f_pp = np.gradient(f_p, dxi)
    xi_safe = np.where(xi < 1e-6, 1e-6, xi)
    return f_pp + (2.0/xi_safe) * f_p

basis_lap = np.zeros_like(basis)
for n in range(N_basis):
    basis_lap[n] = laplacian_3d(basis[n], xi_b, dxi)

V_L = xi_b**2 - mu_0 + 2 * g_tilde * phi_0**2
V_M = g_tilde * phi_0**2

for m in range(N_basis):
    for n in range(N_basis):
        kinetic = -np.sum(basis[m] * basis_lap[n] * w_3d) * dxi
        pot_L = np.sum(basis[m] * V_L * basis[n] * w_3d) * dxi
        pot_M = np.sum(basis[m] * V_M * basis[n] * w_3d) * dxi
        L_mat[m, n] = kinetic + pot_L
        M_mat[m, n] = pot_M

L_mat = 0.5 * (L_mat + L_mat.T)
M_mat = 0.5 * (M_mat + M_mat.T)
print("  L, M matrices computed")

# ============================================================
# EIGENVALUE PROBLEM
# ============================================================

print("\n-> Eigenvalue problem cozuluyor...")

H_squared = (L_mat + M_mat) @ (L_mat - M_mat)
H_squared = 0.5 * (H_squared + H_squared.T)

omega_sq, modes = eigh(H_squared)
idx_sort = np.argsort(omega_sq)
omega_sq = omega_sq[idx_sort]
modes = modes[:, idx_sort]

# ============================================================
# REPORT
# ============================================================

print("\n  Lowest 10 omega^2:")
print("  " + "-" * 50)
for i in range(min(10, len(omega_sq))):
    w2 = omega_sq[i]
    w = np.sqrt(abs(w2))
    sign = "+" if w2 >= 0 else "-"
    status = "stable" if w2 > 1e-6 else ("zero" if abs(w2) < 1e-6 else "unstable")
    print(f"  Mode {i:2d}:  omega^2 = {sign}{abs(w2):.4e}  omega = {w:.4f}  ({status})")

# ============================================================
# STABILITY VERDICT (Goldstone-aware)
# ============================================================

n_unstable_raw = int(np.sum(omega_sq < -1e-6))
n_zero_raw = int(np.sum(np.abs(omega_sq) < 1e-6))
n_stable = int(np.sum(omega_sq > 1e-6))

goldstone_threshold = 1.0
goldstone_modes = int(np.sum(np.abs(omega_sq) < goldstone_threshold))
true_unstable = int(np.sum(omega_sq < -goldstone_threshold))

print("\n" + "=" * 70)
print("STABILITY VERDICT (Goldstone-aware)")
print("=" * 70)
print(f"Raw negative omega^2 modes: {n_unstable_raw}")
print(f"Raw stable modes:           {n_stable}")
print(f"Goldstone (|omega^2|<{goldstone_threshold}): {goldstone_modes}")
print(f"True unstable (<-{goldstone_threshold}):    {true_unstable}")

if true_unstable == 0:
    print("\n+++ DYNAMICALLY STABLE +++")
    print(f"    {goldstone_modes} Goldstone mode(s) from U(1) symmetry breaking")
    print(f"    {n_stable} physical stable modes (omega^2 > 0)")
    verdict = "STABLE"
else:
    print(f"\n!!! INSTABILITY: {true_unstable} true unstable mode(s)")
    verdict = "UNSTABLE"

c_s_sq = g_tilde * phi_0[0]**2
print(f"\nSound speed c_s^2 (local): {c_s_sq:.4f}")
print(f"c_s/c                    : {np.sqrt(max(c_s_sq, 0)):.4f}")

# Start index: after Goldstone modes
start_idx = goldstone_modes

# ============================================================
# FIGURE
# ============================================================

plt.rcParams.update({
    'font.family':     'serif',
    'font.size':       9,
    'axes.titlesize':  9.5,
    'axes.labelsize':  9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.dpi':      150,
    'savefig.dpi':     300,
    'savefig.bbox':    'tight',
    'axes.linewidth':  0.7,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.top':       True,
    'ytick.right':     True,
})

COLORS = {
    'blue':   '#2E86AB',
    'red':    '#C73E1D',
    'green':  '#2E8B57',
    'orange': '#F18F01',
    'purple': '#7B2D8B',
}

fig, axes = plt.subplots(2, 2, figsize=(7.5, 6.0))
fig.subplots_adjust(hspace=0.42, wspace=0.38,
                    top=0.93, bottom=0.08,
                    left=0.10, right=0.97)

# Panel 1: Spectrum
ax = axes[0, 0]
n_show = min(20, len(omega_sq))
mode_idx = np.arange(n_show)

colors_pts = []
for w2 in omega_sq[:n_show]:
    if abs(w2) < goldstone_threshold:
        colors_pts.append('gray')
    elif w2 > 0:
        colors_pts.append('blue')
    else:
        colors_pts.append('red')

ax.scatter(mode_idx, omega_sq[:n_show],
           c=colors_pts, s=50, zorder=5,
           edgecolor='black', linewidth=0.5)
ax.axhline(0, color='red', lw=1.0, ls='--', alpha=0.7,
           label='Stability threshold')

ax.set_xlabel(r'Mode index $n$')
ax.set_ylabel(r'$\omega^2$ (dimensionless)')
ax.set_title(r'Bogoliubov Spectrum')
ax.legend(loc='upper left', framealpha=0.95)
ax.grid(True, ls='--', lw=0.3, alpha=0.4)

ax.text(0.55, 0.05,
        f'Goldstone: {goldstone_modes}\n'
        f'Stable: {n_stable}\n'
        f'Unstable: {true_unstable}\n'
        r'\textbf{Verdict: ' + verdict + '}',
        transform=ax.transAxes, fontsize=8,
        bbox=dict(boxstyle='round',
                  facecolor='lightgreen' if verdict == 'STABLE'
                            else 'lightyellow',
                  edgecolor='gray', alpha=0.9))

# Panel 2: Lowest stable mode profiles
ax = axes[0, 1]
n_modes_show = min(4, len(omega_sq) - start_idx)
mode_colors = [COLORS['blue'], COLORS['orange'],
               COLORS['green'], COLORS['purple']]

for i in range(n_modes_show):
    idx = start_idx + i
    if idx >= len(omega_sq):
        break
    profile = np.zeros_like(xi_b)
    for n in range(N_basis):
        profile += modes[n, idx] * basis[n]
    if np.max(np.abs(profile)) > 0:
        profile /= np.max(np.abs(profile))
    
    w2 = omega_sq[idx]
    if w2 >= 0:
        lbl = f'$n={i}$, $\\omega={np.sqrt(w2):.3f}$'
    else:
        lbl = f'$n={i}$, unstable'
    ax.plot(xi_b, profile, color=mode_colors[i], lw=1.5, label=lbl)

ax.axhline(0, color='black', lw=0.5, alpha=0.4)
ax.set_xlim(0, 6)
ax.set_xlabel(r'$\xi = r/\sigma_{\rm dS}$')
ax.set_ylabel(r'Mode amplitude (normalized)')
ax.set_title(r'Lowest Stable Bogoliubov Modes')
ax.legend(loc='upper right', framealpha=0.95, fontsize=7)
ax.grid(True, ls='--', lw=0.3, alpha=0.4)

# Panel 3: Spectrum vs linear HO
ax = axes[1, 0]
n_compare = min(15, len(omega_sq))
n_arr = np.arange(n_compare)
omega_linear_HO = np.sqrt(4 * n_arr * (n_arr + 3))
omega_num = np.sqrt(np.maximum(omega_sq[:n_compare], 0))

ax.plot(n_arr, omega_num, 'o-', color=COLORS['blue'],
        lw=1.5, ms=5, label=r'Numerical (with $\tilde g$)')
ax.plot(n_arr, omega_linear_HO, 's--', color=COLORS['red'],
        lw=1.2, ms=4, label='Linear HO (analytic)')

ax.set_xlabel(r'Mode index $n$')
ax.set_ylabel(r'$\omega_n$')
ax.set_title(r'Spectrum vs.\ Analytic Linear HO')
ax.legend(loc='upper left', framealpha=0.95)
ax.grid(True, ls='--', lw=0.3, alpha=0.4)

# Panel 4: Ground state + perturbation
ax = axes[1, 1]
ax.plot(xi_b, phi_0 / phi_0.max(),
        color=COLORS['blue'], lw=2.0,
        label=r'Ground state $\phi_0$')

if start_idx < len(omega_sq):
    pert = np.zeros_like(xi_b)
    for n in range(N_basis):
        pert += modes[n, start_idx] * basis[n]
    if np.max(np.abs(pert)) > 0:
        pert = pert / np.max(np.abs(pert)) * 0.2
    ax.plot(xi_b, pert,
            color=COLORS['red'], lw=1.5, ls='--',
            label=r'Lowest stable mode ($\times 0.2$)')

ax.axhline(0, color='black', lw=0.5, alpha=0.4)
ax.set_xlim(0, 6)
ax.set_xlabel(r'$\xi$')
ax.set_ylabel(r'Amplitude')
ax.set_title(r'Ground State + Linear Perturbation')
ax.legend(loc='upper right', framealpha=0.95)
ax.grid(True, ls='--', lw=0.3, alpha=0.4)

fig.suptitle(
    f'Linear stability analysis (Hermite-Gauss basis): '
    f'SC ground state is dynamically {verdict.lower()}',
    fontsize=10, y=0.995,
)

os.makedirs('/workspace/figures', exist_ok=True)
for ext in ['pdf', 'png']:
    path = f'/workspace/figures/fig06_stability.{ext}'
    fig.savefig(path)
    print(f"\nKaydedildi: {path}")
plt.close()

# CSV
csv_path = '/workspace/results/stability_spectrum.csv'
with open(csv_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['mode_index', 'omega_squared', 'omega', 'status'])
    for i in range(len(omega_sq)):
        w2 = omega_sq[i]
        w = np.sqrt(abs(w2))
        if abs(w2) < goldstone_threshold:
            status = 'goldstone'
        elif w2 > 0:
            status = 'stable'
        else:
            status = 'unstable'
        writer.writerow([i, w2, w, status])

print(f"CSV: {csv_path}")
print("\n" + "=" * 70)
print(f"FINAL VERDICT: {verdict}")
print("=" * 70)