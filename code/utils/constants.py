"""
Schwarzschild Condensation Project
Physical constants and unit conversions.

All values from CODATA 2018 / PDG 2023.
"""

import numpy as np
from collections import namedtuple

# ============================================================
# FUNDAMENTAL CONSTANTS (SI)
# ============================================================

# Gravitational constant
G = 6.67430e-11          # m^3 kg^-1 s^-2

# Speed of light
c = 2.99792458e8         # m/s

# Reduced Planck constant
hbar = 1.054571817e-34   # J·s

# Boltzmann constant
k_B = 1.380649e-23       # J/K

# Elementary charge
e_charge = 1.602176634e-19  # C

# Vacuum permittivity
epsilon_0 = 8.8541878128e-12  # F/m

# ============================================================
# PARTICLE MASSES (kg)
# ============================================================

m_electron = 9.1093837015e-31    # kg
m_proton   = 1.67262192369e-27   # kg
m_neutron  = 1.67492749804e-27   # kg

# ============================================================
# PARTICLE MASSES (MeV/c^2)
# ============================================================

m_electron_MeV = 0.51099895      # MeV/c^2
m_proton_MeV   = 938.27208816    # MeV/c^2
m_neutron_MeV  = 939.56542052    # MeV/c^2

# Light quark masses (current, MS-bar at 2 GeV)
m_u_MeV = 2.16                   # MeV
m_d_MeV = 4.67                   # MeV
m_s_MeV = 93.4                   # MeV

# ============================================================
# ASTROPHYSICAL CONSTANTS
# ============================================================

M_sun = 1.98892e30               # kg
R_sun = 6.9634e8                 # m
L_sun = 3.828e26                 # W

# ============================================================
# PLANCK UNITS
# ============================================================

# Planck mass
M_Pl = np.sqrt(hbar * c / G)    # kg
M_Pl_GeV = M_Pl * c**2 / (e_charge * 1e9)  # GeV

# Planck length
l_Pl = np.sqrt(hbar * G / c**3)  # m

# Planck time
t_Pl = np.sqrt(hbar * G / c**5)  # s

# Planck temperature
T_Pl = np.sqrt(hbar * c**5 / (G * k_B**2))  # K

# Planck density
rho_Pl = c**5 / (hbar * G**2)   # kg/m^3

# Planck energy density
epsilon_Pl = rho_Pl * c**2       # J/m^3

# ============================================================
# NUCLEAR PHYSICS
# ============================================================

# Nuclear saturation density
n_0 = 0.16                       # fm^-3
n_0_SI = n_0 * 1e45              # m^-3

# QCD scale
Lambda_QCD_MeV = 217.0           # MeV

# MIT Bag constant (standard)
B_bag_MeV4 = 145.0**4            # MeV^4
B_bag_GeV4 = (145.0e-3)**4       # GeV^4

# ============================================================
# UNIT CONVERSIONS
# ============================================================

MeV_to_J = e_charge * 1e6        # 1 MeV in Joules
GeV_to_J = e_charge * 1e9        # 1 GeV in Joules
fm_to_m  = 1e-15                  # 1 fm in meters
km_to_m  = 1e3                    # 1 km in meters

# hbar*c in various units
hbar_c_MeV_fm = 197.3269804      # MeV·fm
hbar_c_GeV_fm = 0.1973269804     # GeV·fm

# ============================================================
# FINE STRUCTURE CONSTANT
# ============================================================

alpha_em = e_charge**2 / (4 * np.pi * epsilon_0 * hbar * c)
# Should be ≈ 1/137.036

# ============================================================
# SOLAR COMPOSITION (mass fractions)
# ============================================================

SolarElement = namedtuple('SolarElement', 
    ['symbol', 'Z', 'A', 'fraction', 'molar_mass'])

SOLAR_COMPOSITION = [
    SolarElement('H',   1,   1, 0.7346, 1.00794),
    SolarElement('He',  2,   4, 0.2485, 4.00260),
    SolarElement('O',   8,  16, 0.0077, 15.999),
    SolarElement('C',   6,  12, 0.0029, 12.011),
    SolarElement('Fe', 26,  56, 0.0016, 55.845),
    SolarElement('Ne', 10,  20, 0.0012, 20.1797),
    SolarElement('N',   7,  14, 0.0009, 14.0067),
    SolarElement('Si', 14,  28, 0.0007, 28.0855),
    SolarElement('Mg', 12,  24, 0.0005, 24.305),
    SolarElement('S',  16,  32, 0.0004, 32.065),
]

# ============================================================
# VALIDATION
# ============================================================

def validate_constants():
    """Run basic consistency checks on constants."""
    checks = {}
    
    # Fine structure constant
    alpha_check = abs(alpha_em - 1/137.036) / (1/137.036)
    checks['alpha_em'] = alpha_check < 1e-4
    
    # Planck mass
    M_Pl_expected = 2.176434e-8  # kg
    checks['M_Pl'] = abs(M_Pl - M_Pl_expected) / M_Pl_expected < 1e-3
    
    # Planck density
    rho_Pl_expected = 5.155e96  # kg/m^3
    checks['rho_Pl'] = abs(rho_Pl - rho_Pl_expected) / rho_Pl_expected < 0.01
    
    # Solar composition sums to ~1
    total_frac = sum(el.fraction for el in SOLAR_COMPOSITION)
    checks['solar_frac'] = abs(total_frac - 1.0) < 0.01
    
    # Planck temperature
    T_Pl_expected = 1.416784e32  # K
    checks['T_Pl'] = abs(T_Pl - T_Pl_expected) / T_Pl_expected < 1e-3
    
    return checks


if __name__ == '__main__':
    print("=" * 60)
    print("SCHWARZSCHILD CONDENSATION — Physical Constants")
    print("=" * 60)
    print(f"G           = {G:.5e} m^3 kg^-1 s^-2")
    print(f"c           = {c:.8e} m/s")
    print(f"hbar        = {hbar:.6e} J·s")
    print(f"k_B         = {k_B:.6e} J/K")
    print(f"alpha_em    = {alpha_em:.6f} (1/{1/alpha_em:.3f})")
    print(f"M_Pl        = {M_Pl:.4e} kg ({M_Pl_GeV:.4e} GeV)")
    print(f"l_Pl        = {l_Pl:.4e} m")
    print(f"T_Pl        = {T_Pl:.4e} K")
    print(f"rho_Pl      = {rho_Pl:.4e} kg/m^3")
    print(f"M_sun       = {M_sun:.5e} kg")
    print(f"m_neutron   = {m_neutron:.6e} kg")
    print(f"n_0 (nuc.)  = {n_0} fm^-3")
    print()
    
    checks = validate_constants()
    print("Validation:")
    for key, passed in checks.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {key:15s}: {status}")