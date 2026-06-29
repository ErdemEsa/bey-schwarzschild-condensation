"""
Schwarzschild Condensation — Figür 2 (DÜZELTİLMİŞ)
Faz geçiş diyagramı: yoğunluk ekseninde madde fazları
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

G    = C.G
c    = C.c
hbar = C.hbar
k_B  = C.k_B
M_sun  = C.M_sun
rho_Pl = C.rho_Pl
n_0    = C.n_0
m_n    = getattr(C, 'm_n', getattr(C, 'm_neutron'))
m_n_MeV = getattr(C, 'm_n_MeV', getattr(C, 'm_neutron_MeV'))

# ============================================================
# Hesaplama
# ============================================================

def compute_rho(M_kg):
    Rs  = 2 * G * M_kg / c**2
    return M_kg / ((4/3) * np.pi * Rs**3)

def mu_q_from_rho(rho):
    nB     = rho / m_n
    pF     = hbar * (3 * np.pi**2 * nB)**(1/3)
    pF_MeV = pF * c / 1.602176634e-13
    mu_N   = np.sqrt(pF_MeV**2 + m_n_MeV**2)
    return mu_N / 3.0

# ============================================================
# Yoğunluk ekseni
# ============================================================

rho_range  = np.logspace(-4, 32, 3000)
mu_q_range = np.array([mu_q_from_rho(r) for r in rho_range])

# ============================================================
# Faz sınırları
# ============================================================

rho_WD       = 1e9
rho_NS       = 2.3e17
rho_deconf   = 5e17
rho_CFL      = 1e20

# Schwarzschild yoğunlukları
rho_SC_1Msun = compute_rho(1.0 * M_sun)
rho_SC_28    = compute_rho(28.0 * M_sun)
rho_SC_100   = compute_rho(100.0 * M_sun)

# ============================================================
# Matplotlib ayarları
# ============================================================

plt.rcParams.update({
    'font.family':     'serif',
    'font.size':       9,
    'axes.labelsize':  9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'savefig.dpi':     300,
    'savefig.bbox':    'tight',
    'axes.linewidth':  0.7,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.top':       True,
    'ytick.right':     True,
})

# Renkler
c_normal  = '#DDEEFF'
c_WD      = '#BBD5EE'
c_NS      = '#88AADD'
c_quark   = '#FFDDAA'
c_CFL     = '#FFAA77'

# ============================================================
# FIGURE
# ============================================================

fig, (ax_top, ax_bot) = plt.subplots(
    2, 1, figsize=(7.5, 5.5),
    gridspec_kw={'height_ratios': [3.0, 0.9]},
    sharex=True,
)
fig.subplots_adjust(hspace=0.04)

# Eksen sınırları
X_MIN = 1e-4
X_MAX = 1e32

# ──────────────────────────────────────────────
# ÜST PANEL: μq vs rho
# ──────────────────────────────────────────────
ax = ax_top
ax.set_xscale('log')

# Faz arka planları
phase_regions = [
    (X_MIN,      rho_WD,     c_normal),
    (rho_WD,     rho_NS,     c_WD),
    (rho_NS,     rho_CFL,    c_quark),
    (rho_CFL,    X_MAX,      c_CFL),
]

for x0, x1, color in phase_regions:
    ax.axvspan(x0, x1, alpha=0.35, color=color, zorder=1)

# μq eğrisi
ax.plot(rho_range, mu_q_range,
        color='#1A1A8C', lw=2.2, zorder=5,
        label=r'$\mu_q(\bar\rho)$')

# ΛQCD çizgisi
Lambda_QCD = 217.0
ax.axhline(Lambda_QCD, color='#C73E1D', lw=1.3,
           ls='--', zorder=4,
           label=r'$\Lambda_{\rm QCD} = 217\;\mathrm{MeV}$')

# Dekonfineman bölgesi
ax.fill_between(
    rho_range, Lambda_QCD, mu_q_range,
    where=(mu_q_range > Lambda_QCD),
    alpha=0.15, color='#2E8B57',
    label='Deconfinement region',
    zorder=3,
)

# Phoenix A için ek hesap
rho_SC_PhxA   = compute_rho(1e11 * M_sun)
rho_SC_M87    = compute_rho(6.5e9 * M_sun)
rho_SC_SgrA   = compute_rho(4.3e6 * M_sun)

# (rho_değeri, etiket, renk, çizgi_stili, ypos_frac, label_x_konumu)
# label_x_konumu doğrudan x-eksen değeri (logaritmik)
sc_points = [
    (rho_SC_PhxA,  r'Phoenix A',        '#7B2D8B', ':',  0.93, 3e-3),
    (rho_SC_M87,   r'M87$^*$',          '#2D6A4F', '-.', 0.82, 3e0),
    (rho_SC_SgrA,  r'Sgr A$^*$',        '#0077B6', ':',  0.71, 3e6),
    (rho_SC_28,    r'$28\,M_\odot$',    '#1D3557', '--', 0.60, 1e14),
    (rho_SC_1Msun, r'$1\,M_\odot$',     '#E63946', '-',  0.93, 5e20),
]

for rho_val, label, color, ls, ypos_frac, label_x in sc_points:
    ax.axvline(rho_val, color=color, lw=1.3, ls=ls,
               alpha=0.90, zorder=6)
    # Etiketi sabit x konumuna yerleştir, orta hizalı
    ax.text(label_x, ypos_frac * 800,
            label, color=color, fontsize=8,
            ha='center', va='center',
            fontweight='bold',
            zorder=10,
            bbox=dict(boxstyle='round,pad=0.25',
                      facecolor='white',
                      edgecolor=color,
                      lw=0.7,
                      alpha=0.95))
    # Dikey çizgiden etikete bağlantı çizgisi (opsiyonel)
    # Etiket dikey çizginin solundaysa, dikey çizgiden etikete kısa bir çizgi çiziyoruz

ax.set_xlim(X_MIN, X_MAX)
ax.set_ylim(0, 800)
ax.set_ylabel(r'$\mu_q\;[\mathrm{MeV}]$', fontsize=10)
ax.yaxis.set_minor_locator(ticker.MultipleLocator(25))
ax.grid(True, which='major', ls='--', lw=0.3, alpha=0.4)
ax.tick_params(labelbottom=False)

ax.legend(fontsize=8, loc='upper center',
          framealpha=0.95, edgecolor='gray')

ax.set_title(
    r'Phase diagram: quark chemical potential vs.\ mean density at $R_s$',
    fontsize=10, pad=8,
)

# ──────────────────────────────────────────────
# ALT PANEL: Faz etiketleri (yatay çubuk)
# ──────────────────────────────────────────────
ax2 = ax_bot
ax2.set_xscale('log')

phase_labels = [
    (X_MIN,      rho_WD,      c_normal, 'Normal\nmatter'),
    (rho_WD,     rho_NS,      c_WD,     'WD / NS'),
    (rho_NS,     rho_CFL,     c_quark,  'Hadronic / Quark\nmatter'),
    (rho_CFL,    X_MAX,       c_CFL,    'CFL / Color SC\n(BEC core)'),
]

for x0, x1, color, label in phase_labels:
    ax2.axvspan(x0, x1, alpha=0.55, color=color)
    # Log skaladaki geometrik orta nokta
    xmid = np.sqrt(x0 * x1)
    ax2.text(xmid, 0.5, label,
             ha='center', va='center',
             fontsize=8.5,
             fontweight='bold',
             color='#222222')

# Schwarzschild dikey çizgiler
for rho_val, label, color, ls, _, _ in sc_points:
    ax2.axvline(rho_val, color=color, lw=1.3, ls=ls, alpha=0.90)

ax2.set_xlim(X_MIN, X_MAX)
ax2.set_ylim(0, 1)
ax2.set_yticks([])
ax2.set_xlabel(r'$\bar\rho\;[\mathrm{kg\,m^{-3}}]$', fontsize=10)

# Log tikleri görünür yap
ax2.xaxis.set_major_locator(ticker.LogLocator(base=10, numticks=15))
ax2.xaxis.set_minor_locator(
    ticker.LogLocator(base=10, subs='all', numticks=15)
)
ax2.xaxis.set_major_formatter(
    ticker.LogFormatterMathtext(base=10)
)

ax2.grid(True, which='major', axis='x', ls='--', lw=0.3, alpha=0.4)
ax2.tick_params(axis='x', which='major', length=5, top=False)
ax2.tick_params(axis='x', which='minor', length=2.5, top=False)

# ──────────────────────────────────────────────
# Kaydet
# ──────────────────────────────────────────────
os.makedirs('/workspace/figures', exist_ok=True)

for ext in ['pdf', 'png']:
    path = f'/workspace/figures/fig02_phase_diagram.{ext}'
    fig.savefig(path, bbox_inches='tight', dpi=300)
    print(f"Kaydedildi: {path}")

plt.close()
print("Tamamlandı.")