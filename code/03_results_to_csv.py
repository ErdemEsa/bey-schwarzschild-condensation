"""
Schwarzschild Condensation — Sonuçları CSV ve LaTeX tablosuna yaz
Astrofiziksel kara delik adaylarıyla genişletilmiş.
"""

import numpy as np
import sys
import os
import csv

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


def compute_all(M_kg):
    Rs      = 2 * G * M_kg / c**2
    rho     = M_kg / ((4/3) * np.pi * Rs**3)
    nB      = rho / m_n
    pF      = hbar * (3 * np.pi**2 * nB)**(1/3)
    pF_MeV  = pF * c / 1.602176634e-13
    m_n_J   = m_n * c**2
    EF_J    = np.sqrt((pF * c)**2 + m_n_J**2) - m_n_J
    EF_MeV  = EF_J / 1.602176634e-13
    TF      = EF_J / k_B
    mu_N_MeV = np.sqrt(pF_MeV**2 + m_n_MeV**2)
    mu_q_MeV = mu_N_MeV / 3.0
    nB_fm3   = nB * 1e-45
    Lambda_QCD = 217.0
    return {
        'M_Msun':       M_kg / M_sun,
        'Rs_km':        Rs / 1e3,
        'rho':          rho,
        'rho_rho_Pl':   rho / rho_Pl,
        'nB_fm3':       nB_fm3,
        'nB_over_n0':   nB_fm3 / n_0,
        'pF_MeV':       pF_MeV,
        'EF_MeV':       EF_MeV,
        'EF_GeV':       EF_MeV / 1000,
        'TF_K':         TF,
        'mu_N_MeV':     mu_N_MeV,
        'mu_q_MeV':     mu_q_MeV,
        'mu_q_GeV':     mu_q_MeV / 1000,
        'deconf':       mu_q_MeV > Lambda_QCD,
        'mu_q_Lambda':  mu_q_MeV / Lambda_QCD,
    }


# Gerçek + tipik adaylar
kutleler = [
    (1.0,    r'$1\,M_\odot$',                  'Theoretical min.'),
    (3.0,    r'$3\,M_\odot$',                  'TOV upper limit'),
    (10.0,   r'$10\,M_\odot$',                 'Cyg X-1'),
    (28.0,   r'$28\,M_\odot$',                 'GW150914-class'),
    (62.0,   r'$62\,M_\odot$',                 'GW150914 remnant'),
    (1e3,    r'$10^{3}\,M_\odot$',             'IMBH'),
    (4.3e6,  r'$4.3\times 10^{6}\,M_\odot$',   r'Sgr A$^*$'),
    (6.5e9,  r'$6.5\times 10^{9}\,M_\odot$',   r'M87$^*$'),
    (4e10,   r'$4\times 10^{10}\,M_\odot$',    'TON 618'),
    (1e11,   r'$10^{11}\,M_\odot$',            'Phoenix A (cand.)'),
]

sonuclar = []
for M_s, label, source in kutleler:
    r = compute_all(M_s * M_sun)
    r['label']  = label
    r['source'] = source
    sonuclar.append(r)

# ──────────────────────────────────────────────
# CSV kaydet
# ──────────────────────────────────────────────
os.makedirs('/workspace/results', exist_ok=True)
csv_path = '/workspace/results/sc_results.csv'

with open(csv_path, 'w', newline='') as f:
    fieldnames = [k for k in sonuclar[0].keys()
                  if k not in ('label', 'source')]
    fieldnames = ['source'] + fieldnames
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for r in sonuclar:
        row = {k: v for k, v in r.items() if k != 'label'}
        writer.writerow(row)

print(f"CSV kaydedildi: {csv_path}")

# ──────────────────────────────────────────────
# LaTeX tablosu
# ──────────────────────────────────────────────
latex_path = '/workspace/results/table_main.tex'

header = r"""\begin{table*}
\caption{Physical quantities at the Schwarzschild radius for
black holes spanning the full observed mass range, from
stellar-mass remnants to the most massive ultramassive candidate
(Phoenix A). Here $R_s$ is the Schwarzschild radius, $\bar\rho$
the mean density, $n_B/n_0$ the baryon number density in units of
nuclear saturation density, $E_F$ the relativistic Fermi energy,
$T_F$ the Fermi temperature, $\mu_q$ the quark chemical potential,
and the final column indicates whether quark deconfinement is
expected ($\mu_q > \Lambda_{\rm QCD} \simeq 217$ MeV).}
\label{tab:main_results}
\centering
\small
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}
  l l r r r r r c}
\hline\noalign{\smallskip}
$M$ &
Source &
$R_s$ &
$\bar\rho$ &
$n_B/n_0$ &
$E_F$ &
$\mu_q$ &
Deconf. \\
& & [km] & [kg\,m$^{-3}$] & & [GeV] & [MeV] & \\
\noalign{\smallskip}\hline\noalign{\smallskip}
"""

rows = ""
for r in sonuclar:
    deconf_str = r"Yes" if r['deconf'] else r"No"

    rho_exp  = int(np.floor(np.log10(r['rho'])))
    rho_mant = r['rho'] / 10**rho_exp

    nB_exp  = int(np.floor(np.log10(r['nB_over_n0']))) if r['nB_over_n0']>0 else 0
    nB_mant = r['nB_over_n0'] / 10**nB_exp if r['nB_over_n0']>0 else 0

    row = (
        f"{r['label']} & "
        f"{r['source']} & "
        f"{r['Rs_km']:.2f} & "
        f"${rho_mant:.2f}\\times10^{{{rho_exp}}}$ & "
        f"${nB_mant:.2f}\\times10^{{{nB_exp}}}$ & "
        f"{r['EF_GeV']:.2e} & "
        f"{r['mu_q_MeV']:.1f} & "
        f"{deconf_str} \\\\\n"
    )
    rows += row

footer = r"""\noalign{\smallskip}\hline
\end{tabular*}
\end{table*}
"""

with open(latex_path, 'w') as f:
    f.write(header + rows + footer)

print(f"LaTeX tablosu kaydedildi: {latex_path}")

# ──────────────────────────────────────────────
# Terminal özet
# ──────────────────────────────────────────────
print()
print("=" * 100)
print("ÖZET — Tüm gözlenen kara delik kütle aralığı")
print("=" * 100)
print(f"{'Kütle':>16} {'Kaynak':>22} {'Rs(km)':>14} "
      f"{'ρ(kg/m³)':>14} {'μq(MeV)':>10} {'Dekonf.':>10}")
print("-" * 100)
for r in sonuclar:
    d = "EVET ✓" if r['deconf'] else "HAYIR ✗"
    print(
        f"{r['M_Msun']:>14.2e}  "
        f"{r['source']:>22}  "
        f"{r['Rs_km']:>14.3e}  "
        f"{r['rho']:>14.3e}  "
        f"{r['mu_q_MeV']:>10.1f}  "
        f"{d:>10}"
    )
print("=" * 100)