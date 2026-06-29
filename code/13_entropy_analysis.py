"""
Schwarzschild Condensation - Entropy Analysis
Comparison with Bekenstein-Hawking entropy

Computes:
- Number of condensate excitation modes
- Hilbert space dimension
- Entropy from mode counting
- Comparison with S_BH = A/(4 l_Pl^2)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys, os, csv

sys.path.insert(0, '/workspace/code')
from utils import constants as C

G, c, hbar, k_B, M_sun = C.G, C.c, C.hbar, C.k_B, C.M_sun
l_Pl = np.sqrt(hbar * G / c**3)

print("=" * 70)
print("ENTROPY ANALYSIS")
print("=" * 70)
print(f"Planck length: l_Pl = {l_Pl:.3e} m")

# ============================================================
# Mode counting in harmonic trap
# ============================================================

# 3D isotropic harmonic oscillator levels:
# E_{n,l,m} = ℏω(2n + l + 3/2)
# Degeneracy at level N = (N+1)(N+2)/2 where N = 2n + l

m_pair_MeV = 10.0
m_pair_kg = m_pair_MeV * 1e6 * 1.602176634e-19 / c**2

def omega_dS(M_kg):
    """de Sitter harmonic frequency: omega = c/ell = c/R_s"""
    Rs = 2 * G * M_kg / c**2
    return c / Rs

def E_planck():
    """Planck energy in Joules."""
    return np.sqrt(hbar * c**5 / G)

def N_modes_below_Planck(M_kg):
    """
    Number of HO modes with energy E < E_Planck.
    
    E_N = ℏω(N + 3/2), where N = 0, 1, 2, ...
    Each level has degeneracy g(N) = (N+1)(N+2)/2
    
    N_max from: ℏω N_max ≈ E_Planck
    """
    omega = omega_dS(M_kg)
    E_max = E_planck()
    N_max = int(E_max / (hbar * omega))
    
    # Total modes: sum of g(N) for N = 0 to N_max
    # = sum (N+1)(N+2)/2 ≈ N_max^3 / 6 for large N_max
    if N_max < 10:
        total = sum((n+1)*(n+2)//2 for n in range(N_max + 1))
    else:
        total = N_max**3 // 6
    
    return total, N_max

def hilbert_dim_SC(M_kg):
    """
    Approximate Hilbert space dimension:
    dim H ~ (R_s / σ_pair)^3 ~ (M_BH / m_pair)^(3/2)
    Heuristic: many bosons in many modes
    """
    Rs = 2 * G * M_kg / c**2
    omega = c / Rs
    sigma_pair = np.sqrt(hbar / (m_pair_kg * omega))
    return (Rs / sigma_pair)**3

def S_SC(M_kg):
    """Entropy from mode counting: S = k_B ln(dim H)"""
    return k_B * np.log(hilbert_dim_SC(M_kg))

def S_BH(M_kg):
    """Bekenstein-Hawking entropy: S = k_B A / (4 l_Pl^2)"""
    Rs = 2 * G * M_kg / c**2
    A = 4 * np.pi * Rs**2
    return k_B * A / (4 * l_Pl**2)

# ============================================================
# Comparison across mass range
# ============================================================

masses_Msun = [1, 10, 60, 1e3, 1e6, 1e11]
mass_labels = [r'$1\,M_\odot$', r'$10\,M_\odot$', r'$60\,M_\odot$',
               r'$10^3\,M_\odot$', r'$10^6\,M_\odot$',
               r'$10^{11}\,M_\odot$']

print(f"\n{'Mass':>14}  {'dim H_SC':>12}  {'S_SC/k_B':>12}  "
      f"{'S_BH/k_B':>14}  {'ratio':>10}")
print("-" * 75)

results = []
for M_Msun, label in zip(masses_Msun, mass_labels):
    M_kg = M_Msun * M_sun
    dim_H = hilbert_dim_SC(M_kg)
    S_sc = S_SC(M_kg) / k_B
    S_bh = S_BH(M_kg) / k_B
    ratio = S_sc / S_bh
    
    label_clean = label.replace('$', '').replace('\\,', ' ').replace('\\odot', '☉')
    print(f"{label_clean:>14}  {dim_H:>12.3e}  {S_sc:>12.3e}  "
          f"{S_bh:>14.3e}  {ratio:>10.3e}")
    
    results.append({
        'M_Msun': M_Msun,
        'label': label,
        'dim_H_SC': dim_H,
        'S_SC_over_kB': S_sc,
        'S_BH_over_kB': S_bh,
        'ratio': ratio,
    })

print()
print("=" * 70)
print("PHYSICAL INTERPRETATION")
print("=" * 70)

ratio_solar = results[0]['ratio']
print(f"\nFor 1 M_sun:")
print(f"  S_SC / S_BH = {ratio_solar:.3e}")

if ratio_solar < 1:
    print(f"  → SC entropy is {1/ratio_solar:.1e}× smaller than S_BH")
    print(f"  Most BH entropy must come from horizon modes (membrane paradigm)")
    print(f"  SC core contributes only a fraction; remainder is horizon dof")
else:
    print(f"  → SC entropy exceeds S_BH (would violate Bekenstein bound)")

# Scaling
print(f"\nScaling: S_BH ~ M², S_SC ~ M^(3/2)")
print("  Ratio S_SC/S_BH ~ M^(-1/2) → small BHs have relatively more SC entropy")

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
})

COLORS = {
    'blue':   '#2E86AB',
    'red':    '#C73E1D',
    'green':  '#2E8B57',
    'orange': '#F18F01',
}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.5, 3.5))
fig.subplots_adjust(wspace=0.35, top=0.90, bottom=0.15,
                    left=0.10, right=0.97)

M_arr_Msun = np.logspace(-0.5, 12, 100)
M_arr_kg = M_arr_Msun * M_sun

S_SC_arr = np.array([S_SC(M)/k_B for M in M_arr_kg])
S_BH_arr = np.array([S_BH(M)/k_B for M in M_arr_kg])
ratio_arr = S_SC_arr / S_BH_arr

# ─── Panel 1: Both entropies ───
ax1.loglog(M_arr_Msun, S_BH_arr,
           color=COLORS['blue'], lw=2.0,
           label=r'$S_{\rm BH} = A/(4\ell_{\rm Pl}^2)$')
ax1.loglog(M_arr_Msun, S_SC_arr,
           color=COLORS['red'], lw=2.0, ls='--',
           label=r'$S_{\rm SC} \sim \ln(\dim\mathcal{H}_{\rm in})$')

for r in results:
    ax1.scatter(r['M_Msun'], r['S_BH_over_kB'],
                color=COLORS['blue'], s=30, zorder=5,
                edgecolor='black', linewidth=0.5)

ax1.set_xlabel(r'$M\;[M_\odot]$')
ax1.set_ylabel(r'$S / k_B$')
ax1.set_title(r'Entropy: SC vs.\ Bekenstein--Hawking')
ax1.legend(loc='lower right', framealpha=0.95)
ax1.grid(True, which='major', ls='--', lw=0.3, alpha=0.4)

# ─── Panel 2: Ratio ───
ax2.loglog(M_arr_Msun, ratio_arr,
           color=COLORS['green'], lw=2.5)
ax2.axhline(1.0, color='red', lw=1.0, ls='--', alpha=0.7,
            label='S_SC = S_BH')

for r in results:
    ax2.scatter(r['M_Msun'], r['ratio'],
                color=COLORS['green'], s=40, zorder=5,
                edgecolor='black', linewidth=0.5)

ax2.set_xlabel(r'$M\;[M_\odot]$')
ax2.set_ylabel(r'$S_{\rm SC} / S_{\rm BH}$')
ax2.set_title(r'SC Core Entropy Fraction')
ax2.legend(loc='upper right', framealpha=0.95)
ax2.grid(True, which='major', ls='--', lw=0.3, alpha=0.4)

fig.suptitle(
    r'Entropy comparison: SC core vs.\ Bekenstein--Hawking',
    fontsize=10, y=1.00,
)

os.makedirs('/workspace/figures', exist_ok=True)
for ext in ['pdf', 'png']:
    path = f'/workspace/figures/fig11_entropy.{ext}'
    fig.savefig(path)
    print(f"\nKaydedildi: {path}")
plt.close()

# CSV
csv_path = '/workspace/results/entropy_analysis.csv'
with open(csv_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['M_Msun', 'dim_H_SC', 'S_SC_kB',
                     'S_BH_kB', 'ratio'])
    for r in results:
        writer.writerow([r['M_Msun'], r['dim_H_SC'],
                        r['S_SC_over_kB'], r['S_BH_over_kB'],
                        r['ratio']])
print(f"CSV: {csv_path}")

print("\n" + "=" * 70)
print("Entropy analysis complete.")
print("=" * 70)