"""
Schwarzschild Condensation - Timescale Analysis
Critical comparison: condensation vs gravitational collapse

Computes:
- Gravitational free-fall time
- Cooper pair formation time
- BEC condensation time
- Comparison across mass range
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import sys, os, csv

sys.path.insert(0, '/workspace/code')
from utils import constants as C

G, c, hbar, k_B, M_sun = C.G, C.c, C.hbar, C.k_B, C.M_sun

print("=" * 70)
print("TIMESCALE ANALYSIS — Formation vs. Collapse")
print("=" * 70)

# ============================================================
# Parameters
# ============================================================

# CFL pairing gap range
Delta_CFL_MeV = 50.0  # Mid-range value
Delta_CFL_J = Delta_CFL_MeV * 1e6 * 1.602176634e-19

m_pair_MeV = 10.0
m_pair_kg = m_pair_MeV * 1e6 * 1.602176634e-19 / c**2

# Masses to analyze
masses_Msun = [1, 10, 30, 60, 100, 1e3, 1e6, 1e11]
mass_labels = [r'$1\,M_\odot$', r'$10\,M_\odot$', r'$30\,M_\odot$',
               r'$60\,M_\odot$', r'$100\,M_\odot$', r'$10^3\,M_\odot$',
               r'$10^6\,M_\odot$', r'$10^{11}\,M_\odot$ (Phoenix A)']

# ============================================================
# Timescale formulas
# ============================================================

def t_freefall(M_kg):
    """Gravitational free-fall time: t_ff ~ sqrt(R_s^3 / GM)"""
    Rs = 2 * G * M_kg / c**2
    return np.sqrt(Rs**3 / (G * M_kg))

def t_pair_CFL():
    """Cooper pair formation time: tau ~ hbar / Delta_CFL"""
    return hbar / Delta_CFL_J

def t_BEC_relaxation(M_kg):
    """
    BEC relaxation time estimate.
    For trapped BEC: t_BEC ~ 1/omega, where omega = c/ell
    Here ell = R_s (matching at horizon)
    """
    Rs = 2 * G * M_kg / c**2
    omega = c / Rs
    return 1.0 / omega

def t_compress(M_kg):
    """
    Gravitational compression time:
    Time for matter to fall from initial radius to R_s
    For typical stellar collapse, ~ free-fall from initial stellar radius
    """
    # Initial stellar radius ~ R_sun for stellar BHs
    # For SMBHs, larger initial reservoir
    return t_freefall(M_kg) * 10  # rough estimate: 10x free-fall

# ============================================================
# Compute timescales
# ============================================================

print(f"\nCFL pairing gap: Delta = {Delta_CFL_MeV} MeV")
print(f"Pair formation: tau_pair = {t_pair_CFL():.3e} s")
print()
print(f"{'Mass':>16}  {'t_ff (s)':>12}  {'t_pair (s)':>12}  "
      f"{'t_BEC (s)':>12}  {'t_pair/t_ff':>14}")
print("-" * 80)

results = []
tau_pair = t_pair_CFL()

for M_Msun, label in zip(masses_Msun, mass_labels):
    M_kg = M_Msun * M_sun
    
    tff = t_freefall(M_kg)
    tBEC = t_BEC_relaxation(M_kg)
    ratio = tau_pair / tff
    
    label_clean = label.replace('$', '').replace('\\,', ' ').replace('\\odot', '☉')
    print(f"{label_clean:>16}  {tff:>12.3e}  {tau_pair:>12.3e}  "
          f"{tBEC:>12.3e}  {ratio:>14.3e}")
    
    results.append({
        'M_Msun': M_Msun,
        'label': label,
        't_ff': tff,
        't_pair': tau_pair,
        't_BEC': tBEC,
        'ratio_pair_ff': ratio,
    })

print()
print("=" * 70)
print("PHYSICAL INTERPRETATION")
print("=" * 70)

# For 1 M_sun, the critical comparison
r1 = results[0]
print(f"\nFor M = 1 M_sun:")
print(f"  Free-fall time:        t_ff   = {r1['t_ff']:.3e} s")
print(f"  Pair formation:        t_pair = {r1['t_pair']:.3e} s")
print(f"  Ratio:                 t_pair/t_ff = {r1['ratio_pair_ff']:.3e}")

if r1['ratio_pair_ff'] < 1:
    n_orders = -np.log10(r1['ratio_pair_ff'])
    print(f"\n→ t_pair << t_ff by {n_orders:.1f} orders of magnitude")
    print("  Cooper pairing is essentially INSTANTANEOUS")
    print("  relative to gravitational collapse.")
    print("  Condensate formation is kinematically allowed.")
else:
    print(f"\n→ WARNING: t_pair > t_ff, condensation may be kinematically blocked!")

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

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.5, 3.5))
fig.subplots_adjust(wspace=0.35, top=0.90, bottom=0.15,
                    left=0.10, right=0.97)

# Continuous mass range
M_arr_Msun = np.logspace(-0.5, 12, 200)
M_arr_kg = M_arr_Msun * M_sun

tff_arr = np.array([t_freefall(M) for M in M_arr_kg])
tBEC_arr = np.array([t_BEC_relaxation(M) for M in M_arr_kg])
tpair_arr = np.full_like(M_arr_Msun, tau_pair)

# ─── Panel 1: All timescales vs mass ───
ax1.loglog(M_arr_Msun, tff_arr,
           color=COLORS['blue'], lw=2.0,
           label=r'$t_{\rm ff}$ (gravitational)')
ax1.loglog(M_arr_Msun, tBEC_arr,
           color=COLORS['green'], lw=2.0, ls='--',
           label=r'$t_{\rm BEC} = 1/\omega$')
ax1.loglog(M_arr_Msun, tpair_arr,
           color=COLORS['red'], lw=2.0, ls=':',
           label=r'$\tau_{\rm pair} = \hbar/\Delta_{\rm CFL}$')

# Mark specific masses
for r in results:
    ax1.scatter(r['M_Msun'], r['t_ff'],
                color=COLORS['blue'], s=30, zorder=5,
                edgecolor='black', linewidth=0.5)

ax1.set_xlim(0.3, 1e12)
ax1.set_xlabel(r'$M\;[M_\odot]$')
ax1.set_ylabel(r'Timescale [s]')
ax1.set_title(r'Characteristic Timescales')
ax1.legend(loc='upper left', framealpha=0.95, fontsize=7)
ax1.grid(True, which='major', ls='--', lw=0.3, alpha=0.4)

# ─── Panel 2: Ratio (this is THE key plot) ───
ratio_arr = tpair_arr / tff_arr

ax2.loglog(M_arr_Msun, ratio_arr,
           color=COLORS['purple'], lw=2.5,
           label=r'$\tau_{\rm pair} / t_{\rm ff}$')

# Critical line at ratio = 1
ax2.axhline(1.0, color='red', lw=1.5, ls='--', alpha=0.7,
            label='Kinematic threshold')

# Shaded "allowed" region
ax2.axhspan(1e-30, 1.0, alpha=0.10, color='green',
            label='Condensation allowed')

# Mark specific masses
for r in results:
    ax2.scatter(r['M_Msun'], r['ratio_pair_ff'],
                color=COLORS['purple'], s=40, zorder=5,
                edgecolor='black', linewidth=0.5)

ax2.set_xlim(0.3, 1e12)
ax2.set_ylim(1e-30, 10)
ax2.set_xlabel(r'$M\;[M_\odot]$')
ax2.set_ylabel(r'$\tau_{\rm pair} / t_{\rm ff}$')
ax2.set_title(r'Cooper Pairing is Instantaneous')
ax2.legend(loc='upper left', framealpha=0.95, fontsize=7)
ax2.grid(True, which='major', ls='--', lw=0.3, alpha=0.4)

fig.suptitle(
    r'Formation kinetics: Cooper pairing vs.\ gravitational collapse',
    fontsize=10, y=1.00,
)

os.makedirs('/workspace/figures', exist_ok=True)
for ext in ['pdf', 'png']:
    path = f'/workspace/figures/fig08_timescales.{ext}'
    fig.savefig(path)
    print(f"\nKaydedildi: {path}")
plt.close()

# ============================================================
# CSV
# ============================================================
csv_path = '/workspace/results/timescales.csv'
with open(csv_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['M_Msun', 't_ff_s', 't_pair_s',
                     't_BEC_s', 'ratio_pair_ff'])
    for r in results:
        writer.writerow([r['M_Msun'], r['t_ff'], r['t_pair'],
                        r['t_BEC'], r['ratio_pair_ff']])
print(f"CSV: {csv_path}")

print("\n" + "=" * 70)
print("CONCLUSION: Cooper pairing >>> gravitational collapse")
print("Condensation is kinematically unobstructed across all scales.")
print("=" * 70)