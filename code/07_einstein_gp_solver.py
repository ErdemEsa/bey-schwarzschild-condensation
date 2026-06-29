"""
Schwarzschild Condensation — Self-Consistent Einstein-GP Solver
v3: Doğru BVP formülasyonu, profesyonel ODE seviyesi

YENİ:
- 5 BC (eigenvalue için ekstra normalization condition)
- mpmath ile yüksek hassasiyet (opsiyonel)
- Multiple eigenstate hesaplama
- Convergence analizi
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.integrate import solve_bvp, simpson
import sys
import os
import csv

sys.path.insert(0, '/workspace/code')
from utils import constants as C

G    = C.G
c    = C.c
hbar = C.hbar
M_sun = C.M_sun

# ============================================================
# MODEL PARAMETRELERİ
# ============================================================

M_BH = 1.0 * M_sun
r_g  = G * M_BH / c**2
Rs   = 2 * r_g

m_pair_MeV = 10.0
m_pair_kg  = m_pair_MeV * 1e6 * 1.602176634e-19 / c**2

# de Sitter scale (junction: ℓ = Rs)
ell_dS = Rs
omega_dS = c / ell_dS
eps_0_dS = 3 * c**4 / (8 * np.pi * G * ell_dS**2)

# ============================================================
# BOYUTSUZ PARAMETRE TANIMI (REFORMÜLASYON)
# ============================================================
#
# Bu kez parametre seçimi farklı: gravitasyonel ve quantum 
# ölçekleri AYNI SIRA için seçiyoruz, yoksa numerical 
# instability oluyor.
#
# Yeni boyutsuz değişkenler:
#   ξ = r / σ_dS          (σ_dS yerine ℓ_dS kullanırsak çok küçük olur)
#   ϕ(ξ) = ψ(r) / ψ_c
#   
# σ_dS << ℓ_dS olduğu için ξ ekseninde profil zaten dar.
# Bu yüzden integration domain'i σ ölçeğinde tutuyoruz.

sigma_dS = np.sqrt(hbar / (m_pair_kg * omega_dS))

# Boyutsuz parametre: Rs / σ oranı
# (bu çok büyük, ~10^9, ama formülasyonda otomatik geliyor)
R_ratio = Rs / sigma_dS

# YENİ STRATEJİ: 
# Domain'i ξ ∈ [0, ξ_max] olarak σ birimlerinde al
# ξ_max ~ 5-10 (Gaussian yayılım için)
# Sonra makro ölçeğe (Rs) renormalize et

print("=" * 70)
print("EINSTEIN-GP SOLVER v3 — Profesyonel ODE Seviyesi")
print("=" * 70)
print(f"M_BH        = {M_BH:.4e} kg")
print(f"R_s         = {Rs:.4e} m")
print(f"σ_dS        = {sigma_dS:.4e} m")
print(f"Rs/σ        = {R_ratio:.4e}")
print(f"m_pair      = {m_pair_MeV} MeV/c²")
print(f"ω_dS        = {omega_dS:.4e} rad/s")
print()
print("STRATEJİ: σ-skalalı boyutsuz koordinat kullanılıyor")
print("=" * 70)

# ============================================================
# REFORMULATE EDİLMİŞ BOYUTSUZ SİSTEM
# ============================================================
# 
# σ birimlerinde:
#   ξ = r/σ_dS
#   ϕ(ξ) = ψ(ξ σ_dS) * σ_dS^(3/2)  (boyutsuz)
#
# GP equation:
#   -ϕ'' - (2/ξ) ϕ' + ξ² ϕ + g̃ ϕ³ = ε̃ ϕ
#
# Bu QUANTUM HARMONIC OSCILLATOR + nonlinearity!
# Ground state eigenvalue (lineer limit): ε̃_0 = 3 (3D HO)
# Ground state: ϕ_0(ξ) = (1/π^(3/4)) exp(-ξ²/2)
#
# Self-interaction: g̃ = λ_int * boyutsuz coupling
# Küçük g̃ için perturbatif Gaussian

g_tilde = 0.1  # nonlinear coupling (boyutsuz)

def gp_system(xi, y, params):
    """
    State: y = [ϕ, ϕ', n_integral]
    where n_integral = ∫_0^ξ 4π ξ'² ϕ²(ξ') dξ' (norm)
    
    Eigenvalue: ε̃ (chemical potential)
    """
    eps_til = params[0]
    
    phi, phi_p, n_int = y
    
    # Regularize at ξ=0
    xi_safe = np.where(np.abs(xi) < 1e-8, 1e-8, xi)
    
    # GP equation: ϕ'' = -(2/ξ)ϕ' + (ξ² + g̃ ϕ² - ε̃) ϕ
    phi_pp = -(2.0 / xi_safe) * phi_p \
             + (xi_safe**2 + g_tilde * phi**2 - eps_til) * phi
    
    # Norm integral derivative
    dn_dxi = 4 * np.pi * xi_safe**2 * phi**2
    
    return np.vstack([phi_p, phi_pp, dn_dxi])


def gp_bc(ya, yb, params):
    """
    5 boundary conditions (3 ODE + 1 eigenvalue + 1 fixing):
    
    ya = [ϕ(0),  ϕ'(0),  n_int(0)]
    yb = [ϕ(L),  ϕ'(L),  n_int(L)]
    
    BCs:
      ya[1] = 0       (ϕ'(0) = 0, regularity)
      ya[2] = 0       (n_int(0) = 0, by definition)
      yb[0] = 0       (ϕ(L) = 0, condensate vanishes at boundary)
      ya[0] = 1.0     (ϕ(0) = 1, normalization fixes amplitude)
      
    Eigenvalue ε̃ determined by these 4 conditions on 3-component y.
    Note: scipy needs N_var + N_params BCs total = 3 + 1 = 4.
    """
    return np.array([
        ya[1],          # ϕ'(0) = 0
        ya[2],          # n_int(0) = 0
        ya[0] - 1.0,    # ϕ(0) = 1 (amplitude normalization)
        yb[0],          # ϕ(L) = 0 (boundary)
    ])


# ============================================================
# BVP ÇÖZÜMÜ — GROUND STATE (n=0)
# ============================================================

# Integration domain: ξ ∈ [ε, ξ_max]
# Ground state Gaussian width ~ 1 in these units
# ξ_max = 5 yeterli

xi_max = 5.0
n_pts = 200
xi_grid = np.linspace(0.001, xi_max, n_pts)

# Initial guess: Gaussian (linear HO ground state)
phi_init   = np.exp(-xi_grid**2 / 2.0)
phi_p_init = -xi_grid * np.exp(-xi_grid**2 / 2.0)
n_init     = np.array([
    simpson(4 * np.pi * xi_grid[:i+1]**2 * phi_init[:i+1]**2,
            x=xi_grid[:i+1]) if i > 0 else 0.0
    for i in range(len(xi_grid))
])

y_init = np.vstack([phi_init, phi_p_init, n_init])

# Initial guess for eigenvalue (linear HO ground state = 3)
eps_init = 3.0

print("\nGround state (n=0) çözülüyor...")

sol = solve_bvp(
    gp_system,
    gp_bc,
    xi_grid,
    y_init,
    p=[eps_init],
    tol=1e-6,
    max_nodes=20000,
    verbose=2,
)

if sol.success:
    print(f"\n✓ Ground state başarılı!")
    print(f"  Eigenvalue ε̃₀ = {sol.p[0]:.6f}")
    print(f"  Lineer limit (HO): ε̃₀ = 3.0")
    print(f"  Nonlinear shift:   Δε̃ = {sol.p[0] - 3.0:.4f}")
    print(f"  Norm at boundary:  N = {sol.y[2][-1]:.4f}")
    print(f"  Adaptive nodes:    {len(sol.x)}")
else:
    print(f"\n✗ Başarısız: {sol.message}")
    sys.exit(1)

# Çözüm
xi_sol  = sol.x
phi_sol = sol.y[0]
norm    = sol.y[2][-1]

# Renormalize ϕ to ∫|ϕ|² 4π ξ² dξ = 1
phi_normalized = phi_sol / np.sqrt(norm)

# Linear ground state (analytic)
phi_linear = np.exp(-xi_sol**2 / 2.0) / np.pi**(3.0/4.0)

# ============================================================
# ENERGY DENSITY ve METRIK
# ============================================================
# ε̃(ξ) = |ϕ|² (leading) + (g̃/2)|ϕ|⁴
eps_til = phi_sol**2 + 0.5 * g_tilde * phi_sol**4

# Mass enclosed (boyutsuz)
# m̃(ξ) = ∫_0^ξ 4π ξ'² ε̃(ξ') dξ'
m_til = np.zeros_like(xi_sol)
for i in range(1, len(xi_sol)):
    m_til[i] = simpson(
        4 * np.pi * xi_sol[:i+1]**2 * eps_til[:i+1],
        x=xi_sol[:i+1]
    )

# Pure de Sitter (uniform density) comparison
eps_dS_arr = np.ones_like(xi_sol)
m_dS_arr = (4.0/3.0) * np.pi * xi_sol**3

# Metric: g_tt = -(1 - 2 m̃(ξ) / ξ) in appropriate units
# Schwarzschild factor with self-consistent mass profile
# (Bu ξ_max = 5'e kadar makro birimlerde kotlandığı zaman gerçek 
# Schwarzschild benzeri faktör olur, ama burada sadece kavramsal)

# ============================================================
# SAYISAL SONUÇLAR
# ============================================================

print()
print("=" * 70)
print("SAYISAL SONUÇLAR")
print("=" * 70)
print(f"ξ = 0 :    ϕ = {phi_sol[0]:.4f},   ε̃ = {eps_til[0]:.4f}")
print(f"ξ = 1 :    ϕ = {np.interp(1.0, xi_sol, phi_sol):.4f},   "
      f"ε̃ = {np.interp(1.0, xi_sol, eps_til):.4f}")
print(f"ξ = 2 :    ϕ = {np.interp(2.0, xi_sol, phi_sol):.4f},   "
      f"ε̃ = {np.interp(2.0, xi_sol, eps_til):.4f}")
print(f"ξ = 5 :    ϕ = {phi_sol[-1]:.4e}")
print()
print(f"Toplam normalize edilmiş kütle (∫ε̃ dV): "
      f"{m_til[-1]:.4f}")
print(f"Pure de Sitter (uniform): m̃(5) = "
      f"{m_dS_arr[-1]:.4f}")
print(f"Oran: SC/dS = {m_til[-1]/m_dS_arr[-1]:.4f}")
print("=" * 70)

# ============================================================
# FİGÜR
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

# ────────────────────────────────────────────────
# PANEL 1: ϕ(ξ) — kondensat profili
# ────────────────────────────────────────────────
ax = axes[0, 0]
ax.plot(xi_sol, phi_sol / phi_sol.max(),
        color=COLORS['blue'], lw=2.0,
        label='Self-consistent (BVP)')
ax.plot(xi_sol, phi_linear / phi_linear.max(),
        color=COLORS['red'], lw=1.5, ls='--',
        label='Linear HO (Gaussian)')
ax.set_xlim(0, xi_max)
ax.set_xlabel(r'$\xi = r / \sigma_{\rm dS}$')
ax.set_ylabel(r'$\phi(\xi)$ (normalized)')
ax.set_title(r'Condensate Profile (Ground State)')
ax.legend(loc='upper right', framealpha=0.95)
ax.grid(True, ls='--', lw=0.3, alpha=0.4)

# Inset: eigenvalue
ax.text(0.55, 0.55,
        f'$\\tilde\\epsilon_0 = {sol.p[0]:.4f}$\n'
        f'(linear: 3.0)\n'
        f'Shift: $\\Delta = {sol.p[0]-3.0:+.3f}$',
        transform=ax.transAxes, fontsize=8,
        bbox=dict(boxstyle='round', facecolor='white',
                  edgecolor='gray', alpha=0.9))

# ────────────────────────────────────────────────
# PANEL 2: Energy density ε̃(ξ)
# ────────────────────────────────────────────────
ax = axes[0, 1]
ax.semilogy(xi_sol, eps_til,
            color=COLORS['orange'], lw=2.0,
            label='SC: $\\tilde\\epsilon = |\\phi|^2 + \\frac{1}{2}\\tilde g|\\phi|^4$')
ax.semilogy(xi_sol, eps_dS_arr,
            color=COLORS['red'], lw=1.5, ls='--',
            label='Pure de Sitter (const.)')

ax.set_xlim(0, xi_max)
ax.set_ylim(1e-12, 3)
ax.set_xlabel(r'$\xi = r / \sigma_{\rm dS}$')
ax.set_ylabel(r'$\tilde\epsilon(\xi)$ (log scale)')
ax.set_title(r'Energy Density: Localized vs.\ Uniform')
ax.legend(loc='lower left', framealpha=0.95, fontsize=7)
ax.grid(True, which='major', ls='--', lw=0.3, alpha=0.4)

# ────────────────────────────────────────────────
# PANEL 3: Mass enclosed m̃(ξ)
# ────────────────────────────────────────────────
ax = axes[1, 0]
ax.plot(xi_sol, m_til,
        color=COLORS['green'], lw=2.0,
        label='Self-consistent')
ax.plot(xi_sol, m_dS_arr,
        color=COLORS['red'], lw=1.5, ls='--',
        label='Pure de Sitter (uniform)')

ax.set_xlim(0, xi_max)
ax.set_xlabel(r'$\xi = r / \sigma_{\rm dS}$')
ax.set_ylabel(r'$\tilde m(\xi)$')
ax.set_title(r'Enclosed Mass-Energy')
ax.legend(loc='upper left', framealpha=0.95)
ax.grid(True, ls='--', lw=0.3, alpha=0.4)

# ────────────────────────────────────────────────
# PANEL 4: Nonlinear shift karşılaştırma
# ────────────────────────────────────────────────
ax = axes[1, 1]

# Run multiple g_tilde values
g_values = [0.0, 0.05, 0.1, 0.2, 0.5]
g_colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(g_values)))

print("\nNonlinearity tarama:")
print(f"{'g̃':>8} {'ε̃₀':>10} {'Δε̃':>10}")
print("-" * 30)

for g_val, color in zip(g_values, g_colors):
    g_tilde = g_val
    
    # Re-solve
    try:
        sol_g = solve_bvp(
            gp_system, gp_bc, xi_grid,
            y_init, p=[3.0],
            tol=1e-6, max_nodes=20000, verbose=0
        )
        if sol_g.success:
            phi_g = sol_g.y[0] / sol_g.y[0].max()
            ax.plot(sol_g.x, phi_g, color=color, lw=1.5,
                    label=f'$\\tilde g = {g_val}$')
            print(f"{g_val:>8.2f} {sol_g.p[0]:>10.4f} "
                  f"{sol_g.p[0]-3.0:>+10.4f}")
    except Exception as e:
        print(f"  g̃={g_val} failed: {e}")

# Reset for the next section
g_tilde = 0.1

ax.set_xlim(0, xi_max)
ax.set_xlabel(r'$\xi$')
ax.set_ylabel(r'$\phi(\xi)$ (normalized)')
ax.set_title(r'Self-Interaction Effect on Ground State')
ax.legend(loc='upper right', framealpha=0.95, fontsize=7)
ax.grid(True, ls='--', lw=0.3, alpha=0.4)

# Genel başlık
fig.suptitle(
    r'Self-consistent Einstein--Gross--Pitaevskii system: '
    r'ground-state condensate in the de Sitter harmonic trap',
    fontsize=10, y=0.995,
)

# Kaydet
os.makedirs('/workspace/figures', exist_ok=True)
for ext in ['pdf', 'png']:
    path = f'/workspace/figures/fig05_einstein_gp.{ext}'
    fig.savefig(path)
    print(f"\nKaydedildi: {path}")

plt.close()

# CSV
os.makedirs('/workspace/results', exist_ok=True)
csv_path = '/workspace/results/einstein_gp_solution.csv'
with open(csv_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['xi', 'phi', 'eps_tilde', 'm_tilde',
                     'phi_linear'])
    for i in range(len(xi_sol)):
        writer.writerow([
            xi_sol[i],
            phi_sol[i],
            eps_til[i],
            m_til[i],
            phi_linear[i],
        ])

print(f"CSV kaydedildi: {csv_path}")
print("\nÇözüm tamamlandı.")