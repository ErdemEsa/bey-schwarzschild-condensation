# code/05_figure_desitter_core.py

"""
Schwarzschild Condensation — Figür 3
Sol panel : Metrik bileşenleri (iç de Sitter + dış Schwarzschild)
Sağ panel : Kretschmann skaleri K(r)
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

G   = C.G
c   = C.c
M_sun = C.M_sun

# ============================================================
# Model parametreleri  (1 M_sun kara delik)
# ============================================================

M_BH   = 1.0 * M_sun          # kg
Rs     = 2 * G * M_BH / c**2  # Schwarzschild yarıçapı (m)

# İç çözüm: de Sitter çekirdeği
# epsilon_0 sabitini de Sitter ell ile ilişkilendir
# Seçim: R0 = Rs (junction yüzey = olay ufku)
R0 = Rs
# ell^2 = 3c^4 / (8 pi G epsilon_0)
# ve R0 <= ell  koşulundan epsilon_0 seç
# epsilon_0 = 3c^4 / (8 pi G ell^2), ell = R0 seçersek
ell = R0   # junction için ell = R0

# Boyutsuz koordinat: x = r / Rs
x_arr  = np.linspace(0.0, 2.5, 5000)
r_arr  = x_arr * Rs

# ──────────────────────────────────────────────
# Metrik fonksiyonları
# ──────────────────────────────────────────────

def g_tt_inner(r):
    """İç de Sitter metrigi  g_tt = -(1 - r^2/ell^2)"""
    return -(1 - (r / ell)**2)

def g_tt_outer(r):
    """Dış Schwarzschild  g_tt = -(1 - Rs/r)"""
    val = np.where(r > 0, -(1 - Rs / r), np.nan)
    return val

def g_rr_inner(r):
    """İç de Sitter  g_rr = (1 - r^2/ell^2)^{-1}"""
    denom = 1 - (r / ell)**2
    return np.where(np.abs(denom) > 1e-8, 1.0 / denom, np.nan)

def g_rr_outer(r):
    """Dış Schwarzschild  g_rr = (1 - Rs/r)^{-1}"""
    denom = np.where(r > 0, 1 - Rs / r, np.nan)
    return np.where(np.abs(denom) > 1e-8, 1.0 / denom, np.nan)

# Kretschmann skaleri
def K_inner():
    """
    de Sitter uzayında K = (8/3) Lambda^2 = sabit
    Lambda = 3 / ell^2
    """
    Lambda = 3.0 / ell**2
    return (8.0 / 3.0) * Lambda**2   # sabit (r'den bağımsız)

def K_outer(r):
    """
    Schwarzschild dışında K = 48 G^2 M^2 / (c^4 r^6)
    Boyutsuz: K * Rs^6 / (48) = (Rs/r)^6
    """
    return 48.0 * (G * M_BH)**2 / (c**4 * r**6)

# ──────────────────────────────────────────────
# Değerleri hesapla
# ──────────────────────────────────────────────

r_inner = r_arr[x_arr <= 1.0]
x_inner = x_arr[x_arr <= 1.0]

r_outer = r_arr[x_arr >= 1.0]
x_outer = x_arr[x_arr >= 1.0]

gtt_in  = g_tt_inner(r_inner)
grr_in  = g_rr_inner(r_inner)

gtt_out = g_tt_outer(r_outer)
grr_out = g_rr_outer(r_outer)

K_in_val  = K_inner()
K_in_arr  = np.full(len(r_inner), K_in_val)

# r=0'ı dışla
r_outer_K = r_outer[x_outer > 1.001]
x_outer_K = x_outer[x_outer > 1.001]
K_out_arr = K_outer(r_outer_K)

# Normalizasyon: K * ell^4 (boyutsuz)
K_norm     = ell**4
K_in_norm  = K_in_arr  * K_norm
K_out_norm = K_out_arr * K_norm

# ──────────────────────────────────────────────
# FIGURE
# ──────────────────────────────────────────────

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

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 3.2))
fig.subplots_adjust(wspace=0.35)

# ──────────────────────────────
# Sol panel: metrik bileşenleri
# ──────────────────────────────
ax = ax1

# İç bölge arka planı
ax.axvspan(0, 1.0, alpha=0.08, color='#2E86AB')
ax.axvspan(1.0, 2.5, alpha=0.05, color='#F18F01')

# g_tt
ax.plot(x_inner, -gtt_in,
        color='#2E86AB', lw=1.6,
        label=r'$-g_{tt}^{\rm (in)}$')
ax.plot(x_outer, -gtt_out,
        color='#2E86AB', lw=1.6, ls='--')

# g_rr
ax.plot(x_inner, grr_in,
        color='#C73E1D', lw=1.6,
        label=r'$g_{rr}^{\rm (in)}$')
ax.plot(x_outer, grr_out,
        color='#C73E1D', lw=1.6, ls='--')

# Junction yüzeyi
ax.axvline(1.0, color='black', lw=1.0,
           ls=':', alpha=0.8,
           label=r'$r = R_s$ (junction)')

# Etiketler
ax.text(0.25, 0.65, 'de Sitter\n(interior)',
        transform=ax.transAxes,
        fontsize=7.5, color='#2E86AB',
        ha='center')
ax.text(0.78, 0.65, 'Schwarzschild\n(exterior)',
        transform=ax.transAxes,
        fontsize=7.5, color='#F18F01',
        ha='center')

ax.set_xlim(0, 2.5)
ax.set_ylim(-0.1, 2.2)
ax.set_xlabel(r'$r / R_s$', fontsize=9)
ax.set_ylabel(r'Metric components', fontsize=9)
ax.set_title(r'Interior de Sitter + Exterior Schwarzschild',
             fontsize=8.5, pad=4)
ax.legend(fontsize=7.5, loc='upper right',
          framealpha=0.9, edgecolor='none')
ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.1))
ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
ax.grid(True, which='major', ls='--', lw=0.3, alpha=0.4)

# ──────────────────────────────
# Sağ panel: Kretschmann skaleri
# ──────────────────────────────
ax = ax2

ax.axvspan(0, 1.0,  alpha=0.08, color='#2E86AB')
ax.axvspan(1.0, 2.5, alpha=0.05, color='#F18F01')

# İç: sabit K (de Sitter)
x_in_full = np.linspace(0, 1.0, 500)
ax.plot(x_in_full,
        np.full(len(x_in_full), K_in_val * K_norm),
        color='#2E86AB', lw=1.8,
        label=r'$K^{\rm (in)} = (8/3)\Lambda^2$ (const.)')

# Dış: Schwarzschild (1/r^6)
ax.semilogy(x_outer_K,
            K_out_norm,
            color='#C73E1D', lw=1.8, ls='--',
            label=r'$K^{\rm (out)} \propto r^{-6}$')

# Junction
ax.axvline(1.0, color='black', lw=1.0, ls=':', alpha=0.8)

ax.set_xlim(0, 2.5)
ax.set_xlabel(r'$r / R_s$', fontsize=9)
ax.set_ylabel(r'$K \cdot \ell^4$  (Kretschmann scalar)',
              fontsize=9)
ax.set_title(r'Curvature regularity: $K$ finite at $r=0$',
             fontsize=8.5, pad=4)
ax.legend(fontsize=7.5, loc='upper right',
          framealpha=0.9, edgecolor='none')
ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.1))
ax.grid(True, which='major', ls='--', lw=0.3, alpha=0.4)

# Klasik tekillik oku (sağ panelde gösterim)
ax.annotate(
    'Classical singularity\n' + r'$K\to\infty$ here',
    xy=(0.02, K_in_val * K_norm * 10),
    xytext=(0.4, K_in_val * K_norm * 100),
    fontsize=6.5,
    color='gray',
    arrowprops=dict(arrowstyle='->', lw=0.6, color='gray'),
)

# ──────────────────────────────────────────────
# Kaydet
# ──────────────────────────────────────────────
os.makedirs('/workspace/figures', exist_ok=True)

for ext in ['pdf', 'png']:
    path = f'/workspace/figures/fig03_desitter_core.{ext}'
    fig.savefig(path, bbox_inches='tight', dpi=300)
    print(f"Kaydedildi: {path}")

plt.close()
print("Tamamlandı.")