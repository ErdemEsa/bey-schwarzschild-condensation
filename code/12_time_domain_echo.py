"""
Schwarzschild Condensation - Time-Domain Echo Simulation
Simulates gravitational wave echoes from SC cavity

Method:
- 1D Regge-Wheeler equation in tortoise coordinates
- Gaussian initial pulse
- Reflective boundary at SC core
- Time evolution → multiple echoes
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.integrate import simpson
import sys, os, csv

sys.path.insert(0, '/workspace/code')
from utils import constants as C

G, c, hbar, M_sun = C.G, C.c, C.hbar, C.M_sun

print("=" * 70)
print("TIME-DOMAIN ECHO SIMULATION")
print("=" * 70)

# ============================================================
# Setup
# ============================================================

M_BH_Msun = 60  # GW150914-class
M_BH_kg = M_BH_Msun * M_sun
M_geom = G * M_BH_kg / c**2  # meters
r_s = 2 * M_geom

# Cavity parameters
delta_over_Rs = 1e-2
R0 = r_s * (1 + delta_over_Rs)

# Tortoise coordinate range
# r* = r + 2M ln(r/2M - 1)
def tortoise(r, M_g):
    """Tortoise coordinate."""
    return r + 2*M_g * np.log(np.abs(r/(2*M_g) - 1))

# Grid in r*
r_star_min = tortoise(R0, M_geom) - 10 * r_s
r_star_max = tortoise(50 * r_s, M_geom)
N_r = 2000
r_star = np.linspace(r_star_min, r_star_max, N_r)
dr_star = r_star[1] - r_star[0]

print(f"\nM_BH = {M_BH_Msun} M_sun")
print(f"r_s = {r_s:.3e} m")
print(f"R_0 = R_s(1 + {delta_over_Rs}) = {R0:.3e} m")
print(f"Tortoise grid: {N_r} points")
print(f"  r*_min = {r_star_min:.3e} m")
print(f"  r*_max = {r_star_max:.3e} m")

# Convert r_star to r (inverse tortoise)
from scipy.optimize import brentq

def inverse_tortoise(rs, M_g):
    """Find r given r*. For r > 2M only."""
    if rs > 100 * M_g:  # asymptotic
        return rs
    try:
        r = brentq(lambda r: tortoise(r, M_g) - rs,
                   2.001*M_g, 200*M_g, xtol=1e-6*M_g)
        return r
    except:
        return rs  # fallback

print("\n→ Converting tortoise to Schwarzschild r...")
r_arr = np.array([inverse_tortoise(rs, M_geom) for rs in r_star])

# ============================================================
# Regge-Wheeler potential
# ============================================================

def RW_potential(r, M_g, l=2, s=2):
    """
    V_RW(r) = (1 - 2M/r) [l(l+1)/r² + (1-s²)·2M/r³]
    """
    f = 1 - 2*M_g/r
    V = f * (l*(l+1)/r**2 + (1 - s**2)*2*M_g/r**3)
    return V

V_array = np.array([RW_potential(r, M_geom) for r in r_arr])

# ============================================================
# Initial Gaussian pulse (far from BH)
# ============================================================

r_star_pulse = tortoise(20 * r_s, M_geom)
pulse_width = 2 * r_s  # in r* units

Psi_0 = np.exp(-(r_star - r_star_pulse)**2 / (2 * pulse_width**2))
Psi_t_0 = np.zeros_like(Psi_0)  # initially at rest

print(f"\nInitial pulse at r* = {r_star_pulse:.3e} m")
print(f"Pulse width = {pulse_width:.3e} m")

# ============================================================
# Time evolution: leapfrog scheme
# ============================================================
#
# d²Ψ/dt² = d²Ψ/dr*² - V(r) Ψ
#
# Reflective BC at SC core: Ψ(r* = r_star_min) = 0

r_star_core = tortoise(R0, M_geom)
core_idx = np.argmin(np.abs(r_star - r_star_core))
print(f"\nSC core at r* = {r_star_core:.3e} m (index {core_idx})")

# Time grid
c_geom = 1.0  # in geometrized units we use r* units / time units
# But we work in physical: dt < dr*/c (CFL)
dt = 0.4 * dr_star / c
T_total = 200 * r_s / c  # total simulation time
N_t = int(T_total / dt)

print(f"\nTime evolution: {N_t} steps, dt = {dt:.3e} s")
print(f"Total time: {T_total:.3e} s = {T_total*1000:.2f} ms")

# Observer location (where we record signal)
r_star_obs = tortoise(30 * r_s, M_geom)
obs_idx = np.argmin(np.abs(r_star - r_star_obs))
print(f"Observer at r* = {r_star_obs:.3e} m (index {obs_idx})")

# Storage
times = []
signals = []

# Leapfrog
Psi = Psi_0.copy()
Psi_old = Psi_0.copy()  # since velocity is zero

print("\n→ Running time evolution...")
for n in range(N_t):
    t = n * dt
    
    # Compute second spatial derivative
    Psi_pp = np.zeros_like(Psi)
    Psi_pp[1:-1] = (Psi[:-2] - 2*Psi[1:-1] + Psi[2:]) / dr_star**2
    
    # Time evolution
    Psi_new = 2*Psi - Psi_old + dt**2 * c**2 * (Psi_pp - V_array * Psi)
    
    # Boundary conditions
    Psi_new[:core_idx+1] = 0.0  # reflective at core
    Psi_new[-1] = Psi_new[-2]   # outflow at infinity (simple)
    
    # Record observer
    times.append(t)
    signals.append(Psi_new[obs_idx])
    
    # Advance
    Psi_old = Psi
    Psi = Psi_new
    
    if n % (N_t // 20) == 0:
        print(f"  Step {n}/{N_t}, t = {t*1000:.2f} ms")

times = np.array(times)
signals = np.array(signals)

# ============================================================
# Analyze echo structure
# ============================================================

print("\n" + "=" * 70)
print("ECHO ANALYSIS")
print("=" * 70)

# Expected echo period: 2 × tortoise distance from core to peak
r_peak_geom = 1.5 * r_s
r_star_peak = tortoise(r_peak_geom, M_geom)
echo_dt_expected = 2 * (r_star_peak - r_star_core) / c

print(f"\nExpected echo period:")
print(f"  Δt_echo ≈ 2(r*_peak - r*_core)/c = {echo_dt_expected*1000:.2f} ms")

# Find peaks in |signal|
from scipy.signal import find_peaks
abs_signal = np.abs(signals)
peaks, _ = find_peaks(abs_signal, height=0.05*abs_signal.max(),
                       distance=int(echo_dt_expected/dt * 0.5))

print(f"\nDetected {len(peaks)} signal peaks:")
for i, p in enumerate(peaks[:6]):
    print(f"  Peak {i}: t = {times[p]*1000:.2f} ms, "
          f"amplitude = {signals[p]:.3e}")

if len(peaks) > 1:
    intervals = np.diff(times[peaks])
    print(f"\nObserved echo intervals: {intervals*1000} ms")
    print(f"Mean interval: {np.mean(intervals)*1000:.2f} ms")

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
}

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.5, 5.5))
fig.subplots_adjust(hspace=0.40, top=0.93, bottom=0.10,
                    left=0.10, right=0.97)

# ─── Panel 1: Full waveform ───
ax1.plot(times * 1000, signals,
         color=COLORS['blue'], lw=1.2)

# Mark detected peaks
for i, p in enumerate(peaks[:6]):
    color = COLORS['red'] if i == 0 else COLORS['green']
    label = 'Initial ringdown' if i == 0 else (f'Echo {i}' if i == 1 else None)
    ax1.scatter(times[p]*1000, signals[p],
                color=color, s=60, zorder=5,
                edgecolor='black', linewidth=0.7,
                label=label)

ax1.set_xlabel(r'Time [ms]')
ax1.set_ylabel(r'$\Psi(t, r_{\rm obs})$ [arb. units]')
ax1.set_title(rf'Time-Domain Waveform: $M = {M_BH_Msun}\,M_\odot$, '
              rf'$\delta/R_s = 10^{{{int(np.log10(delta_over_Rs))}}}$')
ax1.legend(loc='upper right', framealpha=0.95)
ax1.grid(True, ls='--', lw=0.3, alpha=0.4)
ax1.axhline(0, color='black', lw=0.5, alpha=0.5)

# ─── Panel 2: |signal| log scale (echoes clearer) ───
ax2.semilogy(times * 1000, np.abs(signals) + 1e-15,
             color=COLORS['blue'], lw=1.2)

# Expected echo times
for i in range(1, 6):
    t_echo = times[peaks[0]] + i * echo_dt_expected if len(peaks) > 0 else i * echo_dt_expected
    ax2.axvline(t_echo * 1000, color=COLORS['green'],
                lw=0.8, ls=':', alpha=0.7)
    if i == 1:
        ax2.text(t_echo*1000, 0.5, rf'$\Delta t_{{\rm echo}}$',
                  fontsize=7, color=COLORS['green'],
                  rotation=90, va='center')

ax2.set_xlabel(r'Time [ms]')
ax2.set_ylabel(r'$|\Psi(t)|$ [log scale]')
ax2.set_title(r'Echo Structure (log scale highlights repeating signals)')
ax2.grid(True, which='major', ls='--', lw=0.3, alpha=0.4)
ax2.set_ylim(1e-4, 2)

fig.suptitle(
    r'Time-domain gravitational wave echoes from SC cavity',
    fontsize=10, y=0.99,
)

os.makedirs('/workspace/figures', exist_ok=True)
for ext in ['pdf', 'png']:
    path = f'/workspace/figures/fig10_time_domain_echo.{ext}'
    fig.savefig(path)
    print(f"\nKaydedildi: {path}")
plt.close()

# CSV
csv_path = '/workspace/results/echo_waveform.csv'
with open(csv_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['time_s', 'time_ms', 'signal'])
    # Save every 10th point to keep CSV small
    for i in range(0, len(times), 10):
        writer.writerow([times[i], times[i]*1000, signals[i]])
print(f"CSV: {csv_path}")

print("\n" + "=" * 70)
print(f"Echo simulation complete. {len(peaks)} peaks detected.")
print("=" * 70)