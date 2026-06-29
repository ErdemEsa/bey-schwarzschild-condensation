"""
Schwarzschild Condensation - Kerr Extension Analysis
Robustness of SC core under rotation (slow-rotation limit)

Computes:
- Effective condensation criterion for rotating black holes
- Frame-dragging effects on quark chemical potential
- Comparison: Schwarzschild vs Kerr SC core
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys, os, csv

sys.path.insert(0, '/workspace/code')
from utils import constants as C

G, c, hbar, M_sun = C.G, C.c, C.hbar, C.M_sun
m_n = getattr(C, 'm_n', getattr(C, 'm_neutron'))
m_n_MeV = getattr(C, 'm_n_MeV', getattr(C, 'm_neutron_MeV'))

print("=" * 70)
print("KERR EXTENSION — Slow Rotation Limit")
print("=" * 70)

# ============================================================
# Kerr parameters
# ============================================================

def kerr_radii(M_kg, a_star):
    """
    Kerr horizons:
    r_+ = M + sqrt(M^2 - a^2)  (outer)
    r_- = M - sqrt(M^2 - a^2)  (inner)
    
    a_star = a/M (dimensionless spin), 0 <= a_star <= 1
    """
    M_geom = G * M_kg / c**2  # meters
    a = a_star * M_geom
    
    if a_star > 1:
        return None, None  # naked singularity
    
    r_plus = M_geom + np.sqrt(M_geom**2 - a**2)
    r_minus = M_geom - np.sqrt(M_geom**2 - a**2)
    return r_plus, r_minus

def kerr_volume(M_kg, a_star):
    """Approximate Kerr volume inside outer horizon."""
    r_plus, _ = kerr_radii(M_kg, a_star)
    if r_plus is None:
        return None
    # Use spherical approximation; ellipsoidal correction small for a*<<1
    return (4/3) * np.pi * r_plus**3

def kerr_mean_density(M_kg, a_star):
    """Mean density inside Kerr horizon."""
    V = kerr_volume(M_kg, a_star)
    if V is None or V <= 0:
        return None
    return M_kg / V

def compute_mu_q(rho_kg_m3):
    """Quark chemical potential from density."""
    n_B = rho_kg_m3 / m_n
    p_F = hbar * (3 * np.pi**2 * n_B)**(1/3)
    p_F_MeV = p_F * c / 1.602176634e-13
    mu_n = np.sqrt(p_F_MeV**2 + m_n_MeV**2)
    return mu_n / 3.0

# ============================================================
# Analysis: spin dependence of condensation criterion
# ============================================================

M_ref_Msun = 60  # GW150914-class
M_ref_kg = M_ref_Msun * M_sun

print(f"\nReference mass: M = {M_ref_Msun} M_sun (GW150914-class)")
print(f"\n{'a*':>6}  {'r+ (km)':>10}  {'rho (kg/m³)':>14}  "
      f"{'mu_q (MeV)':>12}  {'Deconf?':>10}")
print("-" * 65)

a_star_values = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99]
Lambda_QCD = 217.0

kerr_results = []
for a in a_star_values:
    r_plus, r_minus = kerr_radii(M_ref_kg, a)
    rho = kerr_mean_density(M_ref_kg, a)
    mu_q = compute_mu_q(rho)
    deconf = "YES" if mu_q > Lambda_QCD else "no"
    
    print(f"{a:>6.2f}  {r_plus/1000:>10.3f}  {rho:>14.3e}  "
          f"{mu_q:>12.2f}  {deconf:>10}")
    
    kerr_results.append({
        'a_star': a,
        'r_plus_m': r_plus,
        'rho': rho,
        'mu_q': mu_q,
        'deconfined': mu_q > Lambda_QCD,
    })

# ============================================================
# Slow-rotation perturbation analysis
# ============================================================

print()
print("=" * 70)
print("PERTURBATIVE ANALYSIS (slow-rotation expansion)")
print("=" * 70)

# To leading order in a_star:
# r_+ = R_s (1 - a*^2/4 + O(a*^4))
# rho_avg(a) = rho_0 (1 + 3 a*^2/4 + O(a*^4))
# So mu_q increases slightly with spin (more confined → denser)

print("\nSchwarzschild → Kerr corrections:")
print(f"  r_+(a*)/r_+(0) ≈ 1 - a*²/4 + O(a*⁴)")
print(f"  ρ(a*)/ρ(0)    ≈ 1 + 3a*²/4 + O(a*⁴)")
print()
print("For a* = 0.7 (typical merger remnant):")
print(f"  r+ shrinks by ~{(1 - np.sqrt(1-0.7**2))/2*100:.1f}%")
print(f"  ρ increases by ~{(1/(1-0.7**2)**(3/2) - 1)*100:.1f}%")
print(f"  μ_q increases by ~{(1/(1-0.7**2)**(0.5) - 1)*100:.1f}%")
print()
print("CONCLUSION: SC condensation criterion is REINFORCED")
print("           for rotating black holes (denser → more")
print("           strongly deconfined).")

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
})

COLORS = {
    'blue':   '#2E86AB',
    'red':    '#C73E1D',
    'green':  '#2E8B57',
    'orange': '#F18F01',
}

# Wider canvas and larger inter-panel spacing prevent the two y-axis labels
# from colliding in the middle after LaTeX scales the figure into the paper.
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.2, 3.9))
fig.subplots_adjust(
    wspace=0.72,
    top=0.82,
    bottom=0.18,
    left=0.075,
    right=0.965,
)

# Continuous spin range
a_arr = np.linspace(0.001, 0.999, 200)
rho_arr = np.array([kerr_mean_density(M_ref_kg, a) for a in a_arr])
mu_q_arr = np.array([compute_mu_q(r) for r in rho_arr])
r_plus_arr = np.array([kerr_radii(M_ref_kg, a)[0] for a in a_arr])
Rs_schw = 2 * G * M_ref_kg / c**2

# ─── Panel 1: r+ and ρ vs spin ───
ax1_twin = ax1.twinx()

ax1.plot(a_arr, r_plus_arr / Rs_schw,
         color=COLORS['blue'], lw=2.0,
         label=r'$r_+/R_s$')
ax1.set_xlabel(r'Spin parameter $a_*$')
ax1.set_ylabel(r'$r_+/R_s$', color=COLORS['blue'], labelpad=6)
ax1.tick_params(axis='y', labelcolor=COLORS['blue'])

ax1_twin.semilogy(a_arr, rho_arr / rho_arr[0],
                   color=COLORS['red'], lw=2.0, ls='--',
                   label=r'$\bar\rho/\bar\rho_0$')
ax1_twin.set_ylabel(r'$\bar\rho/\bar\rho_0$',
                    color=COLORS['red'], labelpad=6)
# Keep the twin-axis label close to the left panel so it does not drift into
# the neighbouring panel's y-label.
ax1_twin.yaxis.set_label_coords(1.10, 0.5)
ax1_twin.tick_params(axis='y', labelcolor=COLORS['red'])

ax1.set_title(r'Kerr Geometry: Horizon and Density')
ax1.set_xlim(0, 1)
ax1.grid(True, ls='--', lw=0.3, alpha=0.4)

# ─── Panel 2: μ_q vs spin ───
ax2.plot(a_arr, mu_q_arr,
         color=COLORS['green'], lw=2.5,
         label=rf'$\mu_q(a_*)$, $M={M_ref_Msun}\,M_\odot$')

ax2.axhline(Lambda_QCD, color=COLORS['red'], lw=1.5, ls='--',
            label=r'$\Lambda_{\rm QCD} = 217$ MeV')

ax2.fill_between(a_arr, Lambda_QCD, mu_q_arr,
                  where=(mu_q_arr > Lambda_QCD),
                  alpha=0.15, color='green',
                  label='Deconfinement region')

for r in kerr_results:
    ax2.scatter(r['a_star'], r['mu_q'],
                color=COLORS['green'], s=40, zorder=5,
                edgecolor='black', linewidth=0.5)

ax2.set_xlabel(r'Spin parameter $a_*$')
ax2.set_ylabel(r'$\mu_q$ [MeV]', labelpad=8)
# Pull the right-panel y-label slightly toward its own axis while preserving
# a clean central gutter between panels.
ax2.yaxis.set_label_coords(-0.12, 0.5)
ax2.set_title(r'Deconfinement is Robust Under Rotation')
ax2.legend(loc='lower right', framealpha=0.95, fontsize=7)
ax2.set_xlim(0, 1)
ax2.grid(True, ls='--', lw=0.3, alpha=0.4)

fig.suptitle(
    r'Robustness of SC criterion under rotation (Kerr geometry)',
    fontsize=10.5, y=0.985,
)

os.makedirs('/workspace/figures', exist_ok=True)
os.makedirs('/workspace/results', exist_ok=True)

# Save both the original manuscript filename and a clearly marked fixed copy.
output_basenames = ['fig09_kerr_extension', 'fig09_kerr_extension_fixed']
for ext in ['pdf', 'png']:
    for base in output_basenames:
        path = f'/workspace/figures/{base}.{ext}'
        fig.savefig(path)
        print(f"\nKaydedildi: {path}")
plt.close()

# CSV
csv_path = '/workspace/results/kerr_analysis.csv'
with open(csv_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['a_star', 'r_plus_m', 'rho_kg_m3',
                     'mu_q_MeV', 'deconfined'])
    for r in kerr_results:
        writer.writerow([r['a_star'], r['r_plus_m'], r['rho'],
                        r['mu_q'], r['deconfined']])
print(f"CSV: {csv_path}")

print("\n" + "=" * 70)
print("Kerr extension complete.")
print("=" * 70)