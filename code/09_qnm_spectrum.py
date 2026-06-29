"""
Schwarzschild Condensation - QNM (Quasi-Normal Mode) Analysis
v2 - Doğru birim dönüşümü
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.optimize import minimize_scalar
import sys, os, csv

sys.path.insert(0, '/workspace/code')
from utils import constants as C

G    = C.G
c    = C.c
hbar = C.hbar
M_sun = C.M_sun

# ============================================================
# Parameters
# ============================================================

masses_Msun = [10, 30, 60, 100, 1e3, 1e6]
mass_labels = [r'$10\,M_\odot$', r'$30\,M_\odot$',
               r'$60\,M_\odot$ (GW150914)', r'$100\,M_\odot$',
               r'$10^3\,M_\odot$', r'$10^6\,M_\odot$']

delta_over_Rs_values = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]

l_mode = 2
s_spin = 2

print("=" * 70)
print("QNM SPECTRUM AND GRAVITATIONAL WAVE ECHO PREDICTIONS")
print("=" * 70)

# ============================================================
# Regge-Wheeler potential (Schwarzschild background)
# ============================================================
# M_BH burada GEOMETRIZED birimde [m], yani M_geom = GM/c^2

def schwarzschild_potential(r, M_geom, l=2, s=2):
    """
    Regge-Wheeler potential. r, M_geom both in meters.
    V has units of 1/m^2.
    """
    r_s = 2 * M_geom
    f = 1 - r_s/r
    V = f * (l*(l+1)/r**2 + (1 - s**2)*r_s/r**3)
    return V

def find_potential_peak(M_geom, l=2, s=2):
    r_s = 2 * M_geom
    result = minimize_scalar(
        lambda r: -schwarzschild_potential(r, M_geom, l, s),
        bounds=(1.01*r_s, 5*r_s),
        method='bounded'
    )
    return result.x, -result.fun

def potential_second_derivative(r, M_geom, l=2, s=2, h=1e-4):
    r_s = 2 * M_geom
    dr = h * r_s
    V_p1 = schwarzschild_potential(r + dr, M_geom, l, s)
    V_m1 = schwarzschild_potential(r - dr, M_geom, l, s)
    V_0  = schwarzschild_potential(r, M_geom, l, s)
    return (V_p1 - 2*V_0 + V_m1) / dr**2

# ============================================================
# WKB QNM frequencies (Schutz-Will)
# ============================================================
# omega has units of 1/m in geometrized units

def wkb_qnm(M_geom, n=0, l=2, s=2):
    """
    Returns (omega_R, omega_I) in units of 1/m.
    """
    r_peak, V_peak = find_potential_peak(M_geom, l, s)
    V_pp = potential_second_derivative(r_peak, M_geom, l, s)
    
    omega_R = np.sqrt(V_peak)
    omega_I = (n + 0.5) * np.sqrt(-2 * V_pp) / (2 * omega_R)
    
    return omega_R, omega_I

# ============================================================
# Echo timescale
# ============================================================

def echo_time(M_BH_kg, delta_over_Rs):
    GM_c3 = G * M_BH_kg / c**3
    return 4 * GM_c3 * abs(np.log(delta_over_Rs))

# ============================================================
# Unit conversion
# ============================================================

def omega_to_Hz(omega_per_m):
    """omega [1/m] -> frequency [Hz]"""
    return omega_per_m * c / (2 * np.pi)

# ============================================================
# Compute QNM for all masses
# ============================================================

print("\nQNM frequencies (l=2, s=2, n=0):")
print("-" * 70)
print(f"{'Mass':>22}  {'f_R (Hz)':>12}  {'gamma (Hz)':>12}  "
      f"{'Q-factor':>10}")
print("-" * 70)

qnm_data = []
for M_Msun, label in zip(masses_Msun, mass_labels):
    M_kg = M_Msun * M_sun
    M_geom = G * M_kg / c**2  # meters
    
    omega_R_per_m, omega_I_per_m = wkb_qnm(M_geom)
    
    f_R = omega_to_Hz(omega_R_per_m)
    gamma_Hz = omega_to_Hz(omega_I_per_m)
    Q = f_R / (2 * gamma_Hz) if gamma_Hz > 0 else float('inf')
    
    label_clean = label.replace('$', '').replace('\\,', ' ').replace('\\odot', 'sun')
    
    print(f"{label_clean:>22}  {f_R:>12.4f}  "
          f"{gamma_Hz:>12.4f}  {Q:>10.4f}")
    
    qnm_data.append({
        'M_Msun': M_Msun,
        'label': label,
        'f_R_Hz': f_R,
        'gamma_Hz': gamma_Hz,
        'Q': Q,
    })

# ============================================================
# Echo times
# ============================================================

print("\nEcho timescales [ms]:")
print("-" * 90)
header = f"{'Mass':>22}"
for d in delta_over_Rs_values:
    header += f" | d/Rs={d:.0e}"
print(header)
print("-" * 90)

echo_data = []
for M_Msun, label in zip(masses_Msun, mass_labels):
    M_kg = M_Msun * M_sun
    row = []
    for delta in delta_over_Rs_values:
        dt = echo_time(M_kg, delta)
        row.append(dt * 1000)
    
    label_clean = label.replace('$', '').replace('\\,', ' ').replace('\\odot', 'sun')
    line = f"{label_clean:>22}"
    for t in row:
        line += f" | {t:>10.3e}"
    print(line)
    
    echo_data.append({
        'M_Msun': M_Msun,
        'label': label,
        'echo_times_ms': row,
    })

# ============================================================
# Detectability analysis
# ============================================================

print("\nDetectability analysis (for d/Rs = 1e-3):")
print("-" * 80)

ligo_band = (10, 1000)
et_band = (1, 10000)
ligo_time_res = 1e-3
et_time_res = 0.1e-3

print(f"{'Mass':>22}  {'f_R in LIGO?':>14}  "
      f"{'Echo>LIGO res?':>16}  {'Echo>ET res?':>14}")
print("-" * 80)

for i, M_Msun in enumerate(masses_Msun):
    label_clean = mass_labels[i].replace('$', '').replace('\\,', ' ').replace('\\odot', 'sun')
    f_R = qnm_data[i]['f_R_Hz']
    echo_t = echo_data[i]['echo_times_ms'][2] / 1000
    
    in_ligo = "YES" if ligo_band[0] <= f_R <= ligo_band[1] else "no"
    above_ligo_res = "YES" if echo_t > ligo_time_res else "no"
    above_et_res = "YES" if echo_t > et_time_res else "no"
    
    print(f"{label_clean:>22}  {in_ligo:>14}  "
          f"{above_ligo_res:>16}  {above_et_res:>14}")

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
    'cyan':   '#17A2B8',
}

fig, axes = plt.subplots(2, 2, figsize=(7.5, 6.0))
fig.subplots_adjust(hspace=0.42, wspace=0.38,
                    top=0.93, bottom=0.08,
                    left=0.10, right=0.97)

# ─── Panel 1: Schwarzschild potential ───
ax = axes[0, 0]
M_geom_ref = G * 60 * M_sun / c**2  # 60 M_sun, in meters
r_s = 2 * M_geom_ref

r_array = np.linspace(1.01 * r_s, 8 * r_s, 500)
V_array = np.array([schwarzschild_potential(r, M_geom_ref, l=2, s=2)
                    for r in r_array])
V_normalized = V_array / V_array.max()

ax.plot(r_array / r_s, V_normalized,
        color=COLORS['blue'], lw=2.0,
        label=r'Schwarzschild $V_{\rm RW}(r)$')

r_peak, V_peak = find_potential_peak(M_geom_ref)
ax.scatter([r_peak/r_s], [V_peak/V_array.max()],
           color=COLORS['red'], s=60, zorder=5,
           edgecolor='black', linewidth=0.7,
           label=r'Light ring ($r \simeq 1.5\,r_s$)')

ax.axvline(1.0, color='black', lw=1.0, ls=':', alpha=0.7,
           label=r'Horizon ($r = r_s$)')

R0 = r_s * (1 + 1e-2)
ax.axvline(R0/r_s, color=COLORS['green'], lw=1.5, ls='--',
           label=r'SC core: $R_0 = R_s(1 + 10^{-2})$')

ax.axvspan(R0/r_s, r_peak/r_s, alpha=0.15,
           color=COLORS['orange'], label='Cavity region')

ax.set_xlim(0.9, 5)
ax.set_ylim(-0.05, 1.15)
ax.set_xlabel(r'$r / r_s$')
ax.set_ylabel(r'$V(r) / V_{\rm peak}$')
ax.set_title(r'Effective Potential and Cavity')
ax.legend(loc='upper right', framealpha=0.95, fontsize=7)
ax.grid(True, ls='--', lw=0.3, alpha=0.4)

# ─── Panel 2: QNM frequency vs mass ───
ax = axes[0, 1]
masses_continuous = np.logspace(0, 11, 100)
f_R_continuous = []
for M_s in masses_continuous:
    M_kg = M_s * M_sun
    M_geom = G * M_kg / c**2
    omega_R, _ = wkb_qnm(M_geom)
    f_R_continuous.append(omega_to_Hz(omega_R))
f_R_continuous = np.array(f_R_continuous)

ax.loglog(masses_continuous, f_R_continuous,
          color=COLORS['blue'], lw=1.5,
          label=r'$f_R$ (l=2, n=0)')

ax.axhspan(10, 1000, alpha=0.15, color=COLORS['green'],
           label='LIGO band (10-1000 Hz)')
ax.axhspan(1e-4, 1e-1, alpha=0.10, color=COLORS['purple'],
           label='LISA band (0.1 mHz-0.1 Hz)')

for d in qnm_data:
    ax.scatter(d['M_Msun'], d['f_R_Hz'],
               color=COLORS['red'], s=40, zorder=5,
               edgecolor='black', linewidth=0.5)

ax.set_xlim(1, 1e11)
ax.set_ylim(1e-7, 1e5)
ax.set_xlabel(r'$M\;[M_\odot]$')
ax.set_ylabel(r'QNM frequency $f_R$ [Hz]')
ax.set_title(r'Ringdown Frequency vs.\ Mass')
ax.legend(loc='upper right', framealpha=0.95, fontsize=7)
ax.grid(True, which='major', ls='--', lw=0.3, alpha=0.4)

# ─── Panel 3: Echo timescale ───
ax = axes[1, 0]

delta_arr = np.logspace(-6, -0.5, 100)
sample_masses = [10, 60, 1000]
sample_labels = [r'$10\,M_\odot$', r'$60\,M_\odot$', r'$10^3\,M_\odot$']
sample_colors = [COLORS['blue'], COLORS['red'], COLORS['green']]

for M_Msun, lbl, col in zip(sample_masses, sample_labels, sample_colors):
    M_kg = M_Msun * M_sun
    echo_t = [echo_time(M_kg, d) * 1000 for d in delta_arr]
    ax.loglog(delta_arr, echo_t, lw=1.8, label=lbl, color=col)

ax.axhline(1.0, color='black', lw=0.8, ls='--', alpha=0.7,
           label='LIGO time res. (1 ms)')
ax.axhline(0.1, color='gray', lw=0.8, ls=':', alpha=0.7,
           label='ET time res. (0.1 ms)')

ax.set_xlim(1e-6, 1)
ax.set_xlabel(r'Cavity parameter $\delta / R_s$')
ax.set_ylabel(r'Echo delay $\Delta t_{\rm echo}$ [ms]')
ax.set_title(r'Gravitational Wave Echo Timescale')
ax.legend(loc='upper right', framealpha=0.95, fontsize=7)
ax.grid(True, which='major', ls='--', lw=0.3, alpha=0.4)

# ─── Panel 4: Detectability heatmap ───
ax = axes[1, 1]

masses_2d = np.array([10, 30, 60, 100, 300, 1000])
deltas_2d = np.logspace(-5, -1, 6)

echo_grid = np.zeros((len(masses_2d), len(deltas_2d)))
for i, M_s in enumerate(masses_2d):
    M_kg = M_s * M_sun
    for j, d in enumerate(deltas_2d):
        echo_grid[i, j] = echo_time(M_kg, d) * 1000

im = ax.imshow(np.log10(echo_grid),
               aspect='auto', origin='lower',
               cmap='viridis',
               extent=[np.log10(deltas_2d[0]),
                       np.log10(deltas_2d[-1]),
                       np.log10(masses_2d[0]),
                       np.log10(masses_2d[-1])])

cbar = plt.colorbar(im, ax=ax,
                    label=r'$\log_{10}(\Delta t_{\rm echo}/\mathrm{ms})$')
cbar.ax.tick_params(labelsize=7)

ax.set_xlabel(r'$\log_{10}(\delta/R_s)$')
ax.set_ylabel(r'$\log_{10}(M/M_\odot)$')
ax.set_title(r'Echo Detectability Map')

# ─── Save ───
fig.suptitle(
    r'Gravitational wave signatures of Schwarzschild Condensation: '
    r'QNM spectrum and echo predictions',
    fontsize=10, y=0.995,
)

os.makedirs('/workspace/figures', exist_ok=True)
for ext in ['pdf', 'png']:
    path = f'/workspace/figures/fig07_qnm_spectrum.{ext}'
    fig.savefig(path)
    print(f"\nKaydedildi: {path}")
plt.close()

# ============================================================
# CSV outputs
# ============================================================

csv_path = '/workspace/results/qnm_data.csv'
with open(csv_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['M_Msun', 'f_R_Hz', 'gamma_Hz', 'Q_factor'])
    for d in qnm_data:
        writer.writerow([d['M_Msun'], d['f_R_Hz'],
                        d['gamma_Hz'], d['Q']])

csv_path2 = '/workspace/results/echo_timescales.csv'
with open(csv_path2, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['M_Msun'] +
                    [f'delta_{d:.0e}_ms' for d in delta_over_Rs_values])
    for d in echo_data:
        writer.writerow([d['M_Msun']] + d['echo_times_ms'])

print(f"CSV: {csv_path}")
print(f"CSV: {csv_path2}")

print("\n" + "=" * 70)
print("QNM ANALYSIS COMPLETE")
print("=" * 70)

# Key findings
f_60 = qnm_data[2]['f_R_Hz']
echo_60 = echo_data[2]['echo_times_ms'][2]
f_1M = qnm_data[5]['f_R_Hz']

print("\nKey findings:")
print(f"  GW150914-like (60 Msun): f_R = {f_60:.1f} Hz (in LIGO band)")
print(f"  Echo delay for d/Rs=1e-3: {echo_60:.2f} ms (LIGO detectable)")
print(f"  SMBH (10^6 Msun): f_R = {f_1M*1000:.2f} mHz (LISA band)")