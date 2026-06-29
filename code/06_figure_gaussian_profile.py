"""
Schwarzschild Condensation — Figür 4
Gaussian Condensate Profile in Harmonic Trap

4 panel:
  - Sol üst : Klasik tekillik vs BEC profil (yarıçap bağımlı)
  - Sağ üst : Harmonic potansiyel V(r) = (1/2)mω²r²
  - Sol alt : Farklı kütleler için BEC profilleri (overlay)
  - Sağ alt : Karakteristik genişlik σ vs kütle
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import sys
import os

sys.path.insert(0, '/workspace/code')
from utils import constants as C

G      = C.G
c      = C.c
hbar   = C.hbar
k_B    = C.k_B
M_sun  = C.M_sun
m_n    = getattr(C, 'm_n', getattr(C, 'm_neutron'))

# ============================================================
# Model parametreleri
# ============================================================

# Cooper pair efektif kütlesi (light di-quark)
m_pair_MeV = 10.0                       # MeV/c^2
m_pair_kg  = m_pair_MeV * 1e6 * 1.602176634e-19 / c**2

def schwarzschild_radius(M_kg):
    return 2 * G * M_kg / c**2

def ell_from_Rs(Rs):
    """Junction: l = Rs (de Sitter çekirdek + Schwarzschild dış)"""
    return Rs

def omega_from_ell(ell):
    """Harmonic trap frekansı: ω = c/ℓ"""
    return c / ell

def sigma_from_omega(omega, m=m_pair_kg):
    """Gaussian karakteristik genişliği: σ = √(ℏ/(mω))"""
    return np.sqrt(hbar / (m * omega))

def gaussian_density(r, sigma):
    """|Ψ₀(r)|² — Gaussian density profile (normalized)"""
    norm = 1.0 / (np.pi**1.5 * sigma**3)
    return norm * np.exp(-r**2 / sigma**2)

def harmonic_potential(r, m, omega):
    """V(r) = (1/2) m ω² r²"""
    return 0.5 * m * omega**2 * r**2

# ============================================================
# Matplotlib ayarları
# ============================================================

plt.rcParams.update({
    'font.family':       'serif',
    'font.size':         9,
    'axes.titlesize':    9,
    'axes.labelsize':    9,
    'xtick.labelsize':   8,
    'ytick.labelsize':   8,
    'legend.fontsize':   7.5,
    'figure.dpi':        150,
    'savefig.dpi':       300,
    'savefig.bbox':      'tight',
    'axes.linewidth':    0.7,
    'xtick.direction':   'in',
    'ytick.direction':   'in',
    'xtick.top':         True,
    'ytick.right':       True,
})

COLORS = {
    'blue':   '#2E86AB',
    'red':    '#C73E1D',
    'green':  '#2E8B57',
    'orange': '#F18F01',
    'purple': '#7B2D8B',
    'dark':   '#1A1A2E',
}

# ============================================================
# FIGURE
# ============================================================

fig, axes = plt.subplots(2, 2, figsize=(7.5, 6.0))
fig.subplots_adjust(hspace=0.45, wspace=0.38,
                    top=0.93, bottom=0.08,
                    left=0.10, right=0.96)

# ────────────────────────────────────────────────────────────
# PANEL 1 (sol üst): Klasik tekillik vs BEC profili
# ────────────────────────────────────────────────────────────
ax = axes[0, 0]

# 1 M_sun için Gaussian
M_BH    = 1.0 * M_sun
Rs_1    = schwarzschild_radius(M_BH)
ell_1   = ell_from_Rs(Rs_1)
omega_1 = omega_from_ell(ell_1)
sigma_1 = sigma_from_omega(omega_1)

# r ekseni — σ küçük olduğundan x'i σ'ya göre ölçekle
# Bu paneli görsel olarak temsili yapıyoruz, gerçek σ çok küçük
sigma_visual = 0.15 * Rs_1   # görsel temsil için
x_arr = np.linspace(0.001, 1.2, 1000)
r_arr = x_arr * Rs_1

# Gaussian profil (görsel)
psi2 = np.exp(-r_arr**2 / sigma_visual**2)
psi2_norm = psi2 / psi2.max()

# Klasik tekillik (δ benzeri, görsel olarak ~1/r³ kesilmiş)
# Sadece kavramsal vurgu için
classical = np.where(x_arr < 0.05, 1.0 / (x_arr / 0.05 + 0.01)**2, 0)
classical_norm = classical / 50  # küçült

ax.plot(x_arr, classical_norm,
        color=COLORS['red'], lw=2.2, ls='--',
        label='Classical (singular)')
ax.fill_between(x_arr, 0, classical_norm,
                color=COLORS['red'], alpha=0.15)

ax.plot(x_arr, psi2_norm,
        color=COLORS['blue'], lw=2.2,
        label=r'SC Gaussian $|\Psi_0(r)|^2$')
ax.fill_between(x_arr, 0, psi2_norm,
                color=COLORS['blue'], alpha=0.20)

# Tekillik vurgusu
ax.annotate(r'$\rho \to \infty$',
            xy=(0.02, 0.95), xycoords='data',
            fontsize=8.5, color=COLORS['red'],
            ha='left', fontweight='bold')

# Düzenli yapı vurgusu
ax.annotate('Finite,\nsmooth,\nregular',
            xy=(0.5, 0.4), xycoords='data',
            fontsize=8, color=COLORS['blue'],
            ha='center')

ax.set_xlim(0, 1.0)
ax.set_ylim(0, 1.1)
ax.set_xlabel(r'$r / R_s$')
ax.set_ylabel(r'Density (normalized)')
ax.set_title(r'Singular vs.\ Regular Core', pad=4)
ax.legend(loc='upper right', framealpha=0.95)
ax.grid(True, ls='--', lw=0.3, alpha=0.4)

# ────────────────────────────────────────────────────────────
# PANEL 2 (sağ üst): Harmonic trap potansiyeli
# ────────────────────────────────────────────────────────────
ax = axes[0, 1]

# Boyutsuz potansiyel: V(r)/(1/2 m ω² Rs²) = (r/Rs)²
x_pot = np.linspace(-1.0, 1.0, 1000)
V_norm = x_pot**2

ax.plot(x_pot, V_norm,
        color=COLORS['orange'], lw=2.2,
        label=r'$V(r) = (1/2)\, m \omega^2 r^2$')
ax.fill_between(x_pot, 0, V_norm,
                color=COLORS['orange'], alpha=0.15)

# Gaussian dalga fonksiyonu (üst üste bindir, küçük ölçekle)
x_gauss = np.linspace(-1.0, 1.0, 1000)
sigma_dim = 0.30  # boyutsuz görsel için
gauss_vis = 0.4 * np.exp(-x_gauss**2 / (2*sigma_dim**2))

ax.plot(x_gauss, gauss_vis,
        color=COLORS['blue'], lw=2.0,
        label=r'$|\Psi_0(r)|^2 \propto e^{-r^2/\sigma^2}$')
ax.fill_between(x_gauss, 0, gauss_vis,
                color=COLORS['blue'], alpha=0.30)

# σ etiketi
ax.annotate('', xy=(sigma_dim, 0.05),
            xytext=(-sigma_dim, 0.05),
            arrowprops=dict(arrowstyle='<->', color=COLORS['dark']))
ax.text(0, 0.10, r'$\sim 2\sigma$',
        ha='center', fontsize=8,
        color=COLORS['dark'], fontweight='bold')

ax.set_xlim(-1.0, 1.0)
ax.set_ylim(0, 1.05)
ax.set_xlabel(r'$r / R_s$')
ax.set_ylabel(r'Potential / Ground state')
ax.set_title(r'Harmonic Trap in de Sitter Interior', pad=4)
ax.legend(loc='upper center', framealpha=0.95, fontsize=7)
ax.grid(True, ls='--', lw=0.3, alpha=0.4)

# ────────────────────────────────────────────────────────────
# PANEL 3 (sol alt): Farklı kütleler için BEC profilleri
# ────────────────────────────────────────────────────────────
ax = axes[1, 0]

masses = [
    (1.0,    r'$1\,M_\odot$',      COLORS['red']),
    (28.0,   r'$28\,M_\odot$',     COLORS['orange']),
    (1e6,    r'$10^6\,M_\odot$',   COLORS['blue']),
    (1e9,    r'$10^9\,M_\odot$',   COLORS['green']),
]

for M_s, label, color in masses:
    Rs = schwarzschild_radius(M_s * M_sun)
    ell = ell_from_Rs(Rs)
    om  = omega_from_ell(ell)
    sg  = sigma_from_omega(om)
    
    sigma_over_Rs = sg / Rs
    log_ratio = int(np.floor(np.log10(sigma_over_Rs)))
    
    # σ birimlerinde standart Gaussian
    x_in_sigma = np.linspace(-4, 4, 1000)
    psi2 = np.exp(-x_in_sigma**2)
    psi2_n = psi2 / psi2.max()
    
    ax.plot(x_in_sigma, psi2_n,
            color=color, lw=1.8,
            label=label + f'  ($\\sigma/R_s \\sim 10^{{{log_ratio}}}$)')

ax.set_xlim(-4, 4)
ax.set_ylim(0, 1.1)
ax.set_xlabel(r'$r / \sigma$')
ax.set_ylabel(r'$|\Psi_0(r)|^2$ (normalized)')
ax.set_title(r'BEC Profile Shape (universal in $\sigma$ units)', pad=4)
ax.legend(loc='upper right', framealpha=0.95, fontsize=6.5)
ax.grid(True, ls='--', lw=0.3, alpha=0.4)

# ────────────────────────────────────────────────────────────
# PANEL 4 (sağ alt): σ vs M (BEC core ölçeği)
# ────────────────────────────────────────────────────────────
ax = axes[1, 1]

M_sun_range = np.logspace(-0.5, 12, 200)
M_kg_range  = M_sun_range * M_sun

sigma_arr = np.zeros(len(M_kg_range))
Rs_arr    = np.zeros(len(M_kg_range))

for i, M in enumerate(M_kg_range):
    Rs = schwarzschild_radius(M)
    ell = ell_from_Rs(Rs)
    om  = omega_from_ell(ell)
    sigma_arr[i] = sigma_from_omega(om)
    Rs_arr[i]    = Rs

# σ kendisi (m cinsinden)
ax.loglog(M_sun_range, sigma_arr,
          color=COLORS['blue'], lw=1.8,
          label=r'$\sigma$ (BEC core size)')

# Karşılaştırma: Rs
ax.loglog(M_sun_range, Rs_arr,
          color=COLORS['red'], lw=1.8, ls='--',
          label=r'$R_s$ (Schwarzschild)')

# Planck length referansı
l_Pl = np.sqrt(hbar * G / c**3)
ax.axhline(l_Pl, color='black', lw=0.7, ls=':', alpha=0.6)
ax.text(2e10, l_Pl * 3, r'$\ell_{\rm Pl}$',
        fontsize=7, color='black')

# Astrofiziksel noktalar
sp_masses = [1.0, 28.0, 4.3e6, 1e11]
sp_labels = [r'$1\,M_\odot$', r'$28\,M_\odot$', 'Sgr A$^*$', 'Phoenix A']
sp_colors = [COLORS['red'], COLORS['orange'],
             COLORS['green'], COLORS['purple']]

for M_s, lbl, col in zip(sp_masses, sp_labels, sp_colors):
    Rs = schwarzschild_radius(M_s * M_sun)
    ell = ell_from_Rs(Rs)
    om  = omega_from_ell(ell)
    sg  = sigma_from_omega(om)
    ax.scatter(M_s, sg, color=col, s=30, zorder=5,
               edgecolor='white', linewidth=0.7)

ax.set_xlim(3e-1, 1e12)
ax.set_xlabel(r'$M\;[M_\odot]$')
ax.set_ylabel(r'Length scale [m]')
ax.set_title(r'Core Size $\sigma$ vs.\ Black Hole Mass', pad=4)
ax.legend(loc='upper left', framealpha=0.95)
ax.xaxis.set_major_locator(ticker.LogLocator(base=10, numticks=10))
ax.yaxis.set_major_locator(ticker.LogLocator(base=10, numticks=10))
ax.xaxis.set_minor_locator(ticker.NullLocator())
ax.yaxis.set_minor_locator(ticker.NullLocator())
ax.grid(True, which='major', ls='--', lw=0.3, alpha=0.4)

# ────────────────────────────────────────────────────────────
# Genel başlık
# ────────────────────────────────────────────────────────────
fig.suptitle(
    r'Gaussian condensate profile in the de Sitter harmonic trap',
    fontsize=10, y=0.995,
)

# ────────────────────────────────────────────────────────────
# Kaydet
# ────────────────────────────────────────────────────────────
os.makedirs('/workspace/figures', exist_ok=True)
for ext in ['pdf', 'png']:
    path = f'/workspace/figures/fig04_gaussian_profile.{ext}'
    fig.savefig(path)
    print(f"Kaydedildi: {path}")

plt.close()

# ────────────────────────────────────────────────────────────
# Bilgi çıktısı
# ────────────────────────────────────────────────────────────
print()
print("=" * 60)
print("BEC PROFIL ÖZELLİKLERİ")
print("=" * 60)

for M_s, lbl, _ in zip(sp_masses, sp_labels, sp_colors):
    Rs = schwarzschild_radius(M_s * M_sun)
    ell = ell_from_Rs(Rs)
    om  = omega_from_ell(ell)
    sg  = sigma_from_omega(om)
    print(f"{lbl:>20}:  Rs = {Rs:.3e} m,  σ = {sg:.3e} m,  σ/Rs = {sg/Rs:.3e}")

print("=" * 60)
print(f"Cooper pair mass:     m_pair = {m_pair_MeV} MeV/c²")
print(f"Planck length:        l_Pl   = {l_Pl:.3e} m")
print("=" * 60)