"""
Schwarzschild Condensation — Figür 1 (GENİŞLETİLMİŞ)
Kara delik kütlesi vs fiziksel büyüklükler (4 panel)
Kütle aralığı: 1 M_sun → 10^11 M_sun (Phoenix A)
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
rho_Pl = C.rho_Pl
n_0    = C.n_0
m_n    = getattr(C, 'm_n', getattr(C, 'm_neutron'))
m_n_MeV = getattr(C, 'm_n_MeV', getattr(C, 'm_neutron_MeV'))

# ============================================================
# Hesaplama
# ============================================================

def compute_all(M_kg):
    Rs   = 2 * G * M_kg / c**2
    rho  = M_kg / ((4/3) * np.pi * Rs**3)
    nB   = rho / m_n
    pF   = hbar * (3 * np.pi**2 * nB)**(1/3)
    pF_MeV = pF * c / 1.602176634e-13
    m_n_J  = m_n * c**2
    EF_J   = np.sqrt((pF * c)**2 + m_n_J**2) - m_n_J
    EF_MeV = EF_J / 1.602176634e-13
    TF     = EF_J / k_B
    mu_N_MeV = np.sqrt(pF_MeV**2 + m_n_MeV**2)
    mu_q_MeV = mu_N_MeV / 3.0
    nB_fm3   = nB * 1e-45
    return {
        'Rs_km':      Rs / 1e3,
        'rho':        rho,
        'nB_over_n0': nB_fm3 / n_0,
        'TF_K':       TF,
        'mu_q_MeV':   mu_q_MeV,
        'EF_GeV':     EF_MeV / 1000,
    }

# ============================================================
# Kütle tarama: 1 M_sun → 10^12 M_sun
# ============================================================

M_sun_range = np.logspace(-0.5, 12, 1500)
M_kg_range  = M_sun_range * M_sun

Rs_km_arr   = np.zeros(len(M_kg_range))
rho_arr     = np.zeros(len(M_kg_range))
nB_n0_arr   = np.zeros(len(M_kg_range))
mu_q_arr    = np.zeros(len(M_kg_range))

for i, M in enumerate(M_kg_range):
    r = compute_all(M)
    Rs_km_arr[i]  = r['Rs_km']
    rho_arr[i]    = r['rho']
    nB_n0_arr[i]  = r['nB_over_n0']
    mu_q_arr[i]   = r['mu_q_MeV']

# ============================================================
# Matplotlib ayarları
# ============================================================

plt.rcParams.update({
    'font.family':       'serif',
    'font.size':         9,
    'axes.titlesize':    9,
    'axes.labelsize':    9,
    'xtick.labelsize':   7.5,
    'ytick.labelsize':   7.5,
    'legend.fontsize':   7.5,
    'figure.dpi':        150,
    'savefig.dpi':       300,
    'savefig.bbox':      'tight',
    'axes.linewidth':    0.7,
    'xtick.direction':   'in',
    'ytick.direction':   'in',
    'xtick.major.size':  3.5,
    'ytick.major.size':  3.5,
    'xtick.minor.size':  2.0,
    'ytick.minor.size':  2.0,
    'xtick.top':         True,
    'ytick.right':       True,
})

COLORS = {
    'blue':   '#2E86AB',
    'red':    '#C73E1D',
    'green':  '#2E8B57',
    'orange': '#F18F01',
    'purple': '#7B2D8B',
}

# ============================================================
# Özel astrofiziksel noktalar
# ============================================================

# (Kütle [M_sun], etiket, kategori_rengi)
special = [
    (1.0,    r'$1\,M_\odot$',                 '#888888'),
    (28.0,   r'$28\,M_\odot$',                COLORS['red']),
    (4.3e6,  r'Sgr A$^*$',                    COLORS['blue']),
    (6.5e9,  r'M87$^*$',                      COLORS['green']),
    (1e11,   r'Phoenix A',                    COLORS['purple']),
]

sp_results = [(M_s, label, color, compute_all(M_s * M_sun))
              for M_s, label, color in special]

# ============================================================
# 4 PANEL
# ============================================================

fig, axes = plt.subplots(2, 2, figsize=(7.5, 6.0))
fig.subplots_adjust(hspace=0.42, wspace=0.38,
                    top=0.93, bottom=0.08,
                    left=0.10, right=0.97)

X_MIN = 3e-1
X_MAX = 1e12

# ──────────────────────────────────────────────
# Panel 1: Schwarzschild yarıçapı
# ──────────────────────────────────────────────
ax = axes[0, 0]
ax.loglog(M_sun_range, Rs_km_arr,
          color=COLORS['blue'], lw=1.6)

# Özel noktalar
y_offsets = [2.5, 2.5, 0.3, 2.5, 0.3]
x_offsets = [3.5, 3.5, 0.25, 3.5, 0.18]

for i, (M_s, label, color, r) in enumerate(sp_results):
    ax.scatter(M_s, r['Rs_km'],
               color=color, s=30, zorder=6,
               edgecolor='white', linewidth=0.7)
    ax.annotate(
        label,
        xy=(M_s, r['Rs_km']),
        xytext=(M_s * x_offsets[i], r['Rs_km'] * y_offsets[i]),
        fontsize=6.5,
        color=color,
        arrowprops=dict(arrowstyle='->', lw=0.5, color=color),
    )

ax.set_xlim(X_MIN, X_MAX)
ax.set_xlabel(r'$M\;[M_\odot]$')
ax.set_ylabel(r'$R_s\;[\mathrm{km}]$')
ax.set_title(r'Schwarzschild Radius', pad=4)
ax.xaxis.set_major_locator(ticker.LogLocator(base=10, numticks=10))
ax.yaxis.set_major_locator(ticker.LogLocator(base=10, numticks=10))
ax.xaxis.set_minor_locator(ticker.NullLocator())
ax.yaxis.set_minor_locator(ticker.NullLocator())
ax.grid(True, which='major', ls='--', lw=0.3, alpha=0.5)

# ──────────────────────────────────────────────
# Panel 2: Ortalama yoğunluk
# ──────────────────────────────────────────────
ax = axes[0, 1]
ax.loglog(M_sun_range, rho_arr,
          color=COLORS['orange'], lw=1.6)

# ρ_nuc referansı
rho_nuc = 2.3e17
ax.axhline(rho_nuc, color=COLORS['green'],
           lw=0.9, ls='--', alpha=0.8)
ax.text(1.5e0, rho_nuc * 3,
        r'$\rho_{\rm nuc}$', fontsize=7.5, color=COLORS['green'])

# ρ_water referansı (gözünüze yardımcı)
ax.axhline(1e3, color='gray', lw=0.7, ls=':', alpha=0.6)
ax.text(1.5e0, 3e3,
        r'$\rho_{\rm water}$', fontsize=7, color='gray')

for i, (M_s, label, color, r) in enumerate(sp_results):
    ax.scatter(M_s, r['rho'],
               color=color, s=30, zorder=6,
               edgecolor='white', linewidth=0.7)

ax.set_xlim(X_MIN, X_MAX)
ax.set_ylim(1e-7, 1e21)
ax.set_xlabel(r'$M\;[M_\odot]$')
ax.set_ylabel(r'$\bar\rho\;[\mathrm{kg\,m^{-3}}]$')
ax.set_title(r'Mean Density at $R_s$', pad=4)
ax.xaxis.set_major_locator(ticker.LogLocator(base=10, numticks=10))
ax.yaxis.set_major_locator(ticker.LogLocator(base=10, numticks=8))
ax.xaxis.set_minor_locator(ticker.NullLocator())
ax.yaxis.set_minor_locator(ticker.NullLocator())
ax.grid(True, which='major', ls='--', lw=0.3, alpha=0.5)

# ──────────────────────────────────────────────
# Panel 3: Baryon yoğunluğu / n0
# ──────────────────────────────────────────────
ax = axes[1, 0]
ax.loglog(M_sun_range, nB_n0_arr,
          color=COLORS['purple'], lw=1.6)

# n_B = n_0 çizgisi
ax.axhline(1.0, color='gray', lw=0.8, ls='--', alpha=0.8)
ax.text(1.5e0, 2.0,
        r'$n_B = n_0$', fontsize=7, color='gray')

# ~10 n_0 (QCD eşiği)
ax.axhline(10.0, color=COLORS['red'], lw=0.8, ls=':', alpha=0.8)
ax.text(1.5e0, 20.0,
        r'$\sim10\,n_0$ (QCD)', fontsize=7, color=COLORS['red'])

for i, (M_s, label, color, r) in enumerate(sp_results):
    ax.scatter(M_s, r['nB_over_n0'],
               color=color, s=30, zorder=6,
               edgecolor='white', linewidth=0.7)

ax.set_xlim(X_MIN, X_MAX)
ax.set_xlabel(r'$M\;[M_\odot]$')
ax.set_ylabel(r'$n_B / n_0$')
ax.set_title(r'Baryon Density (units of $n_0$)', pad=4)
ax.xaxis.set_major_locator(ticker.LogLocator(base=10, numticks=10))
ax.yaxis.set_major_locator(ticker.LogLocator(base=10, numticks=8))
ax.xaxis.set_minor_locator(ticker.NullLocator())
ax.yaxis.set_minor_locator(ticker.NullLocator())
ax.grid(True, which='major', ls='--', lw=0.3, alpha=0.5)

# ──────────────────────────────────────────────
# Panel 4: Kuark kimyasal potansiyeli
# ──────────────────────────────────────────────
ax = axes[1, 1]
ax.semilogx(M_sun_range, mu_q_arr,
            color=COLORS['green'], lw=1.6)

# ΛQCD
Lambda_QCD = 217.0
ax.axhline(Lambda_QCD, color=COLORS['red'],
           lw=1.0, ls='--', alpha=0.9)
ax.text(1.5e0, Lambda_QCD - 25,
        r'$\Lambda_{\rm QCD} = 217\;\mathrm{MeV}$',
        fontsize=7, color=COLORS['red'])

# Dekonfineman bölgesi
ax.fill_between(
    M_sun_range,
    Lambda_QCD, mu_q_arr,
    where=(mu_q_arr > Lambda_QCD),
    alpha=0.12, color=COLORS['green'],
)

for i, (M_s, label, color, r) in enumerate(sp_results):
    ax.scatter(M_s, r['mu_q_MeV'],
               color=color, s=30, zorder=6,
               edgecolor='white', linewidth=0.7)

ax.set_xlim(X_MIN, X_MAX)
ax.set_ylim(180, 700)
ax.set_xlabel(r'$M\;[M_\odot]$')
ax.set_ylabel(r'$\mu_q\;[\mathrm{MeV}]$')
ax.set_title(r'Quark Chemical Potential', pad=4)
ax.xaxis.set_major_locator(ticker.LogLocator(base=10, numticks=10))
ax.xaxis.set_minor_locator(ticker.NullLocator())
ax.yaxis.set_minor_locator(ticker.MultipleLocator(50))
ax.grid(True, which='major', ls='--', lw=0.3, alpha=0.5)

# ──────────────────────────────────────────────
# Genel başlık
# ──────────────────────────────────────────────
fig.suptitle(
    r'Physical quantities at the Schwarzschild radius vs.\ '
    r'black hole mass ($1\,M_\odot$ to Phoenix A)',
    fontsize=10, y=0.995,
)

# ──────────────────────────────────────────────
# Kaydet
# ──────────────────────────────────────────────
os.makedirs('/workspace/figures', exist_ok=True)
for ext in ['pdf', 'png']:
    filepath = f'/workspace/figures/fig01_mass_density.{ext}'
    fig.savefig(filepath)
    print(f"Kaydedildi: {filepath}")
plt.close()