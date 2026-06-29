"""
Schwarzschild Condensation — Temel Hesaplamalar
1. Schwarzschild yarıçapı
2. Ortalama yoğunluk
3. Baryon sayı yoğunluğu
4. Fermi momentumu
5. Fermi sıcaklığı
"""

import numpy as np
import sys

# /workspace/code klasörünü import yoluna ekle
sys.path.insert(0, '/workspace/code')

# constants modülünü topluca al
from utils import constants as C

# Sabitleri burada eşitleyelim
G = C.G
c = C.c
hbar = C.hbar
k_B = C.k_B
M_sun = C.M_sun
rho_Pl = C.rho_Pl
T_Pl = C.T_Pl
n_0 = C.n_0

# İsim farklarını güvenli şekilde çöz
m_n = getattr(C, 'm_n', getattr(C, 'm_neutron'))
m_n_MeV = getattr(C, 'm_n_MeV', getattr(C, 'm_neutron_MeV'))
hbar_c = getattr(C, 'hbar_c', getattr(C, 'hbar_c_MeV_fm'))
fm_to_m = getattr(C, 'fm_to_m', 1e-15)


def schwarzschild_yaricapi(M_kg):
    """Schwarzschild yarıçapını hesapla."""
    Rs = 2 * G * M_kg / c**2
    return Rs  # metre


def ortalama_yogunluk(M_kg, Rs_m):
    """Schwarzschild hacmindeki ortalama yoğunluk."""
    V = (4/3) * np.pi * Rs_m**3
    rho = M_kg / V
    return rho  # kg/m^3


def baryon_sayi_yogunlugu(rho_kg_m3):
    """Kütle yoğunluğundan baryon sayı yoğunluğu."""
    n_B_SI = rho_kg_m3 / m_n
    n_B_fm3 = n_B_SI * 1e-45
    return n_B_SI, n_B_fm3


def fermi_momentumu_baryon(n_B_SI):
    """Sıfır-sıcaklık Fermi momentumu."""
    pF = hbar * (3 * np.pi**2 * n_B_SI)**(1/3)
    pF_MeV = pF * c / (1.602176634e-13)  # MeV
    return pF, pF_MeV


def fermi_enerjisi(pF_kg_m_s):
    """Rölativistik Fermi enerjisi."""
    m_n_J = m_n * c**2
    EF_J = np.sqrt((pF_kg_m_s * c)**2 + m_n_J**2) - m_n_J
    EF_MeV = EF_J / 1.602176634e-13
    return EF_J, EF_MeV


def fermi_sicakligi(EF_J):
    """Fermi sıcaklığı."""
    TF = EF_J / k_B
    return TF


def quark_kimyasal_potansiyeli(mu_N_MeV):
    """Nötron kimyasal potansiyelinden kuark kimyasal potansiyeli."""
    mu_q_MeV = mu_N_MeV / 3.0
    return mu_q_MeV


def hesapla_tek_kutle(M_kg, etiket=""):
    print(f"\n{'='*60}")
    print(f"  {etiket}  |  M = {M_kg:.4e} kg  |  M = {M_kg/M_sun:.1f} M_sun")
    print(f"{'='*60}")

    Rs = schwarzschild_yaricapi(M_kg)
    print(f"Schwarzschild yarıçapı  Rs = {Rs:.4e} m  = {Rs/1e3:.3f} km")

    rho = ortalama_yogunluk(M_kg, Rs)
    print(f"Ortalama yoğunluk       ρ  = {rho:.4e} kg/m^3")
    print(f"                           = {rho/rho_Pl:.4e} × ρ_Pl")

    n_B_SI, n_B_fm3 = baryon_sayi_yogunlugu(rho)
    print(f"Baryon say. yoğunluğu   nB = {n_B_SI:.4e} m^-3")
    print(f"                           = {n_B_fm3:.4e} fm^-3")
    print(f"                           = {n_B_fm3/n_0:.4e} × n_0")

    pF, pF_MeV = fermi_momentumu_baryon(n_B_SI)
    pF_GeV = pF_MeV / 1000
    print(f"Fermi momentumu         pF = {pF:.4e} kg m/s")
    print(f"                           = {pF_MeV:.4e} MeV/c")
    print(f"                           = {pF_GeV:.4e} GeV/c")

    EF_J, EF_MeV = fermi_enerjisi(pF)
    EF_GeV = EF_MeV / 1000
    EF_TeV = EF_GeV / 1000
    print(f"Fermi enerjisi          EF = {EF_MeV:.4e} MeV")
    print(f"                           = {EF_GeV:.4e} GeV")
    print(f"                           = {EF_TeV:.4e} TeV")

    TF = fermi_sicakligi(EF_J)
    print(f"Fermi sıcaklığı         TF = {TF:.4e} K")
    print(f"                           = {TF/1e16:.4f} × 10^16 K")

    mu_N_MeV = np.sqrt(pF_MeV**2 + m_n_MeV**2)
    mu_q_MeV = quark_kimyasal_potansiyeli(mu_N_MeV)
    mu_q_GeV = mu_q_MeV / 1000
    print(f"Nötron kimyasal pot.    μN = {mu_N_MeV:.4e} MeV")
    print(f"Kuark kimyasal pot.     μq = {mu_q_MeV:.4e} MeV")
    print(f"                           = {mu_q_GeV:.4e} GeV")

    Lambda_QCD = 217.0
    dekonfinement = mu_q_MeV > Lambda_QCD
    print(f"Kuark dekonfinemanı?       {'EVET ✓' if dekonfinement else 'HAYIR ✗'}  "
          f"(μq/ΛQCD = {mu_q_MeV/Lambda_QCD:.2f})")

    return {
        'M_kg': M_kg,
        'M_Msun': M_kg / M_sun,
        'Rs_m': Rs,
        'Rs_km': Rs / 1e3,
        'rho': rho,
        'rho_over_rho_Pl': rho / rho_Pl,
        'n_B_SI': n_B_SI,
        'n_B_fm3': n_B_fm3,
        'n_B_over_n0': n_B_fm3 / n_0,
        'pF_MeV': pF_MeV,
        'EF_MeV': EF_MeV,
        'EF_GeV': EF_GeV,
        'TF_K': TF,
        'mu_N_MeV': mu_N_MeV,
        'mu_q_MeV': mu_q_MeV,
        'mu_q_GeV': mu_q_GeV,
        'dekonfinement': dekonfinement,
    }


def karsilastirma_tablosu(sonuclar):
    print(f"\n\n{'='*90}")
    print("KARŞILAŞTIRMA TABLOSU")
    print(f"{'='*90}")
    print(f"{'Kütle':>12} {'Rs (km)':>10} {'ρ (kg/m³)':>14} {'nB/n0':>12} "
          f"{'EF (GeV)':>12} {'μq (GeV)':>12} {'Dekonfin':>10}")
    print("-" * 90)

    for s in sonuclar:
        dekonfin = "EVET ✓" if s['dekonfinement'] else "HAYIR ✗"
        print(
            f"{s['M_Msun']:>8.0f} M☉  "
            f"{s['Rs_km']:>10.2f}  "
            f"{s['rho']:>14.3e}  "
            f"{s['n_B_over_n0']:>12.3e}  "
            f"{s['EF_GeV']:>12.3e}  "
            f"{s['mu_q_GeV']:>12.3e}  "
            f"{dekonfin:>10}"
        )
    print("=" * 90)


if __name__ == "__main__":
    print("\n" + "█" * 60)
    print("  SCHWARZSCHILD CONDENSATION — ANA HESAPLAMALAR")
    print("█" * 60)

    kutleler = [
        (1.0,   "1 Güneş Kütlesi"),
        (28.0,  "28 Güneş Kütlesi"),
        (100.0, "100 Güneş Kütlesi"),
        (1e6,   "1 Milyon Güneş Kütlesi"),
    ]

    sonuclar = []
    for carpan, etiket in kutleler:
        M = carpan * M_sun
        s = hesapla_tek_kutle(M, etiket)
        sonuclar.append(s)

    karsilastirma_tablosu(sonuclar)

    s1 = sonuclar[0]
    print(f"\n{'='*60}")
    print("EK KONTROLLER (1 M☉)")
    print(f"{'='*60}")
    print(f"Planck yoğunluğuyla oran: ρ/ρ_Pl = {s1['rho_over_rho_Pl']:.4e}")
    print(f"Fermi sıcaklığı:  TF = {s1['TF_K']:.3e} K")
    print(f"Planck sıcaklığı: TP = {T_Pl:.3e} K")
    print(f"TF/TP = {s1['TF_K']/T_Pl:.4e}")