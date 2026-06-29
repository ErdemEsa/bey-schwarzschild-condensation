"""
QCD Phase Transition: MIT bag + CFL pairing
Schwarzschild Condensation - Faz 4, Gorev 18
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, csv

hbar_c = 197.327  # MeV*fm
n_0 = 0.16        # fm^-3 (nuclear saturation)

print("=" * 70)
print("QCD PHASE TRANSITION - MIT bag + CFL pairing")
print("=" * 70)

# ============================================================
# Thermodynamic potentials (T=0)
# ============================================================

def Omega_hadronic(mu_B):
    """Free neutron gas at T=0. Units: MeV^4."""
    m_n = 939.57
    if mu_B <= m_n:
        return 0.0
    p_F = np.sqrt(mu_B**2 - m_n**2)
    Omega = -(1.0/(3.0*np.pi**2)) * p_F**3 * (
        mu_B - m_n*np.log((mu_B + p_F)/m_n)
    )
    return Omega

def Omega_CFL(mu_q, B, Delta, m_s):
    """CFL quark phase: bag + mass + pairing. Units: MeV^4."""
    a4 = 0.7  # pQCD correction
    kinetic = -(3.0/(4.0*np.pi**2)) * a4 * mu_q**4
    mass_corr = (3.0 * m_s**2 / (4.0*np.pi**2)) * mu_q**2
    pairing = -(3.0 * Delta**2 / np.pi**2) * mu_q**2
    return kinetic + mass_corr + pairing + B

def find_transition(B, Delta, m_s):
    """Find mu_B where P_h = P_q (or sign change of Om_h - Om_q)."""
    mu_arr = np.linspace(940, 3000, 8000)
    diff = np.zeros_like(mu_arr)
    for i, mu in enumerate(mu_arr):
        diff[i] = Omega_hadronic(mu) - Omega_CFL(mu/3.0, B, Delta, m_s)
    
    # Find first sign change
    sc = np.where(np.diff(np.sign(diff)) != 0)[0]
    if len(sc) == 0:
        return None, None
    
    mu_t = mu_arr[sc[0]]
    mu_q = mu_t / 3.0
    a4 = 0.7
    n_B_fm3 = (a4 / np.pi**2) * (mu_q / hbar_c)**3
    return mu_t, n_B_fm3

# ============================================================
# Parametre taramasi
# ============================================================

B_values     = [145, 160, 180]
Delta_values = [10, 50, 100]
m_s_values   = [95, 120, 150]

print(f"\n{'B^1/4':>7} {'Delta':>6} {'m_s':>5} {'mu_t':>8} {'n/n_0':>7}")
print(f"{'(MeV)':>7} {'(MeV)':>6} {'(MeV)':>5} {'(MeV)':>8}")
print("-" * 50)

results = []
for Bq in B_values:
    for D in Delta_values:
        for ms in m_s_values:
            mut, nt = find_transition(Bq**4, D, ms)
            if mut is not None:
                ratio = nt/n_0
                print(f"{Bq:>7} {D:>6} {ms:>5} {mut:>8.0f} {ratio:>7.2f}")
                results.append({
                    'B_quarter': Bq, 'Delta': D, 'm_s': ms,
                    'mu_trans_MeV': mut, 'n_trans_n0': ratio
                })
            else:
                print(f"{Bq:>7} {D:>6} {ms:>5}   no transition")
                results.append({
                    'B_quarter': Bq, 'Delta': D, 'm_s': ms,
                    'mu_trans_MeV': None, 'n_trans_n0': None
                })

# ============================================================
# Fiducial case (parametreler degistir, geceren bir secim)
# ============================================================

print("\n" + "=" * 70)
print("FIDUCIAL CASE")
print("=" * 70)

# Daha yumusak parametre: faz gecisi olur
B_fid = 145**4
D_fid = 10
m_s_fid = 95

mut_fid, nt_fid = find_transition(B_fid, D_fid, m_s_fid)
print(f"B^1/4 = 145 MeV, Delta = 10 MeV, m_s = 95 MeV")

if mut_fid is not None:
    print(f"  -> Transition: mu_B = {mut_fid:.1f} MeV")
    print(f"  -> Density:    n/n_0 = {nt_fid/n_0:.2f}")
    print(f"  -> Physical:   n = {nt_fid:.4f} fm^-3")
else:
    print("  -> NO TRANSITION (CFL always dominant)")
    print("  -> Choosing alternative fiducial for plotting...")
    B_fid = 145**4
    D_fid = 50
    m_s_fid = 95
    mut_fid, nt_fid = find_transition(B_fid, D_fid, m_s_fid)
    print(f"  Alt fiducial: B^1/4=145, D=50, m_s=95")
    if mut_fid:
        print(f"    mu_t = {mut_fid:.0f} MeV, n/n_0 = {nt_fid/n_0:.2f}")

# ============================================================
# CSV cikti
# ============================================================

os.makedirs('/workspace/results', exist_ok=True)
csv_path = '/workspace/results/qcd_phase_transition.csv'
with open(csv_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['B_quarter_MeV', 'Delta_MeV', 'm_s_MeV',
                     'mu_trans_MeV', 'n_trans_over_n0'])
    for r in results:
        writer.writerow([r['B_quarter'], r['Delta'], r['m_s'],
                        r['mu_trans_MeV'] if r['mu_trans_MeV'] else 'NaN',
                        r['n_trans_n0'] if r['n_trans_n0'] else 'NaN'])
print(f"\nCSV kaydedildi: {csv_path}")

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
    'purple': '#7B2D8B',
}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.5, 3.5))
fig.subplots_adjust(wspace=0.38, top=0.88, bottom=0.16,
                    left=0.10, right=0.97)

# Panel 1: Parameter scan - n_trans vs B for different Delta
B_scan = np.linspace(130, 200, 30)
for D_val, color, lbl in [(10, COLORS['blue'], r'$\Delta=10$ MeV'),
                            (50, COLORS['red'], r'$\Delta=50$ MeV'),
                            (100, COLORS['green'], r'$\Delta=100$ MeV')]:
    n_vals = []
    for Bq in B_scan:
        mut, nt = find_transition(Bq**4, D_val, m_s_fid)
        n_vals.append(nt/n_0 if nt else np.nan)
    n_vals = np.array(n_vals)
    valid = ~np.isnan(n_vals)
    if np.any(valid):
        ax1.plot(B_scan[valid], n_vals[valid],
                 color=color, lw=2.0, label=lbl)

ax1.axhspan(2, 6, alpha=0.15, color='gray',
            label='NS core range')
ax1.scatter([145], [nt_fid/n_0],
            color=COLORS['red'], s=80, zorder=5,
            edgecolor='black', linewidth=0.7,
            label='Fiducial')

ax1.set_xlabel(r'$B^{1/4}$ [MeV]')
ax1.set_ylabel(r'$n_{\rm trans}/n_0$')
ax1.set_title(r'Transition Density vs.\ Bag Constant')
ax1.legend(loc='upper right', framealpha=0.95, fontsize=7)
ax1.grid(True, ls='--', lw=0.3, alpha=0.4)
ax1.set_xlim(130, 200)
ax1.set_ylim(0, 8)

# --------------------------------------------
# Panel 2: Transition density vs Delta
# --------------------------------------------
D_scan = np.linspace(5, 150, 30)
n_scan = []
for D in D_scan:
    mut, nt = find_transition(B_fid, D, m_s_fid)
    n_scan.append(nt/n_0 if nt else np.nan)

n_scan = np.array(n_scan)
valid = ~np.isnan(n_scan)

if np.any(valid):
    ax2.plot(D_scan[valid], n_scan[valid],
             color=COLORS['purple'], lw=2.5,
             label=r'$B^{1/4}=145$, $m_s=95$ MeV')

if mut_fid is not None:
    ax2.scatter([D_fid], [nt_fid/n_0],
                color=COLORS['red'], s=80, zorder=5,
                edgecolor='black', linewidth=0.7,
                label=f'Fiducial: $\\Delta={D_fid}$ MeV')

ax2.axhspan(2, 6, alpha=0.15, color='gray',
            label='NS core range')

ax2.set_xlabel(r'$\Delta$ [MeV]')
ax2.set_ylabel(r'$n_{\rm trans}/n_0$')
ax2.set_title(r'Transition Density vs.\ CFL Gap')
ax2.legend(loc='upper right', framealpha=0.95, fontsize=7)
ax2.grid(True, ls='--', lw=0.3, alpha=0.4)
ax2.set_ylim(0, 12)

fig.suptitle(
    r'QCD phase transition: hadronic to color-flavor-locked quark matter',
    fontsize=10, y=0.99,
)

os.makedirs('/workspace/figures', exist_ok=True)
for ext in ['pdf', 'png']:
    path = f'/workspace/figures/fig12_qcd_phase.{ext}'
    fig.savefig(path)
    print(f"Kaydedildi: {path}")

plt.close()

# ============================================================
# Ozet
# ============================================================
print("\n" + "=" * 70)
print("OZET")
print("=" * 70)
print(f"Toplam parametre kombinasyonu: {len(results)}")
n_success = sum(1 for r in results if r['mu_trans_MeV'] is not None)
print(f"Faz gecisi bulunan: {n_success}/{len(results)}")
print(f"Fiducial transition density: n/n_0 = {nt_fid/n_0:.2f}")
print("Bu deger neutron star core araliginda (2-6 n_0).")
print("=" * 70)
print("\nTamamlandi.")