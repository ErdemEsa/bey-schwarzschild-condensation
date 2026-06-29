#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
22_einstein_qcd_tov_solver.py

BEY / Schwarzschild Condensation patch-v6:
Einstein + QCD-inspired EOS / isotropic TOV solver.

Purpose
-------
This module upgrades the BEY/SC regular-core program from a pure ansatz
diagnostic to an effective Einstein-system calculation:

    Einstein equations + QCD-inspired CFL/MIT-bag effective EOS
    -> TOV equilibrium profiles
    -> mass-radius-compactness curves
    -> metric-function diagnostics

Boundary
--------
This is NOT a full first-principles Einstein-QCD solution.
It is an Einstein + QCD-inspired effective EOS/TOV calculation.

Units
-----
Internal TOV integration uses geometrized units G=c=1 with length in meters.
Energy density and pressure are converted from MeV/fm^3 to geometric 1/m^2.

EOS
---
A linear QCD-inspired bag/CFL effective EOS is used:

    p = (epsilon - epsilon_surface)/3

where

    epsilon_surface = max(4 B - 3 P_pair, epsilon_floor)

and

    P_pair ~ 3 Delta^2 mu_ref^2 / pi^2 / (hbar c)^3

in MeV/fm^3.

This is a controlled phenomenological EOS, not a microscopic derivation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Physical constants
G = 6.67430e-11
c = 299792458.0
M_sun = 1.98847e30
MEVFM3_TO_JM3 = 1.602176634e32
HBARC_MEV_FM = 197.3269804


@dataclass(frozen=True)
class EOSParams:
    B_MeVfm3: float = 80.0
    Delta_MeV: float = 50.0
    mu_ref_MeV: float = 400.0
    cs2: float = 1.0 / 3.0
    epsilon_floor_MeVfm3: float = 120.0


@dataclass(frozen=True)
class TOVConfig:
    r_max_km: float = 40.0
    dr_m: float = 25.0
    p_surface_frac: float = 1e-8


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def mevfm3_to_geo(x: float | np.ndarray) -> float | np.ndarray:
    """Convert energy density / pressure from MeV/fm^3 to geometric 1/m^2."""
    return G * (np.asarray(x) * MEVFM3_TO_JM3) / c**4


def geo_to_mevfm3(x_geo: float | np.ndarray) -> float | np.ndarray:
    return np.asarray(x_geo) * c**4 / G / MEVFM3_TO_JM3


def mass_m_to_msun(m_geo_m: float | np.ndarray) -> float | np.ndarray:
    return np.asarray(m_geo_m) * c**2 / G / M_sun


def cfl_pair_pressure_MeVfm3(params: EOSParams) -> float:
    # CFL condensation pressure scale in natural units converted to MeV/fm^3.
    p_mev4 = 3.0 * params.Delta_MeV**2 * params.mu_ref_MeV**2 / (math.pi**2)
    return p_mev4 / (HBARC_MEV_FM**3)


def epsilon_surface_MeVfm3(params: EOSParams) -> float:
    p_pair = cfl_pair_pressure_MeVfm3(params)
    eps = 4.0 * params.B_MeVfm3 - 3.0 * p_pair
    return max(eps, params.epsilon_floor_MeVfm3)


def pressure_from_epsilon_MeVfm3(eps: float | np.ndarray, params: EOSParams) -> float | np.ndarray:
    eps0 = epsilon_surface_MeVfm3(params)
    return params.cs2 * (np.asarray(eps) - eps0)


def epsilon_from_pressure_MeVfm3(p: float | np.ndarray, params: EOSParams) -> float | np.ndarray:
    eps0 = epsilon_surface_MeVfm3(params)
    return eps0 + np.asarray(p) / params.cs2


def tov_rhs(r: float, m: float, p: float, params: EOSParams) -> tuple[float, float, float]:
    """Return dm/dr, dp/dr, dPhi/dr in geometric units."""
    if p <= 0:
        return 0.0, 0.0, 0.0
    eps = mevfm3_to_geo(epsilon_from_pressure_MeVfm3(geo_to_mevfm3(p), params))

    # Avoid singular center and horizon denominator issues.
    denom = max(r * (r - 2.0 * m), 1e-30)
    dm = 4.0 * math.pi * r**2 * eps
    dphi = (m + 4.0 * math.pi * r**3 * p) / denom
    dp = -(eps + p) * (m + 4.0 * math.pi * r**3 * p) / denom
    return dm, dp, dphi


def rk4_step(r: float, m: float, p: float, phi: float, h: float, params: EOSParams) -> tuple[float, float, float]:
    def f(rr, mm, pp, ph):
        dm, dp, dph = tov_rhs(rr, mm, pp, params)
        return np.array([dm, dp, dph], dtype=float)

    y = np.array([m, p, phi], dtype=float)
    k1 = f(r, *y)
    k2 = f(r + 0.5*h, *(y + 0.5*h*k1))
    k3 = f(r + 0.5*h, *(y + 0.5*h*k2))
    k4 = f(r + h, *(y + h*k3))
    yn = y + (h/6.0) * (k1 + 2*k2 + 2*k3 + k4)
    return float(yn[0]), float(yn[1]), float(yn[2])


def integrate_star(eps_c_MeVfm3: float, params: EOSParams, cfg: TOVConfig) -> dict[str, object]:
    p_c_MeVfm3 = pressure_from_epsilon_MeVfm3(eps_c_MeVfm3, params)
    if p_c_MeVfm3 <= 0:
        raise ValueError("Central energy density must be above EOS surface density.")

    dr = cfg.dr_m
    r = dr
    eps_c_geo = mevfm3_to_geo(eps_c_MeVfm3)
    p_geo = mevfm3_to_geo(p_c_MeVfm3)
    m = 4.0 * math.pi * r**3 * eps_c_geo / 3.0
    phi = 0.0

    rows = []
    p_floor = max(p_geo * cfg.p_surface_frac, mevfm3_to_geo(1e-8))
    r_max = cfg.r_max_km * 1000.0

    while r < r_max and p_geo > p_floor and r > 2.0*m:
        eps_geo = mevfm3_to_geo(epsilon_from_pressure_MeVfm3(geo_to_mevfm3(p_geo), params))
        f_metric = 1.0 - 2.0*m/r
        rows.append({
            "r_km": r/1000.0,
            "m_msun": float(mass_m_to_msun(m)),
            "epsilon_MeVfm3": float(geo_to_mevfm3(eps_geo)),
            "p_MeVfm3": float(geo_to_mevfm3(p_geo)),
            "phi_raw": phi,
            "f_metric": f_metric,
            "compactness_2m_r": 2.0*m/r,
        })
        m_new, p_new, phi_new = rk4_step(r, m, p_geo, phi, dr, params)
        r += dr
        m, p_geo, phi = m_new, p_new, phi_new

        if not np.isfinite(p_geo) or not np.isfinite(m) or not np.isfinite(phi):
            break

    prof = pd.DataFrame(rows)
    if prof.empty:
        raise RuntimeError("Integration produced empty profile.")

    R_m = float(prof["r_km"].iloc[-1] * 1000.0)
    M_m = float(prof["m_msun"].iloc[-1] * G * M_sun / c**2)
    C = 2.0 * M_m / R_m

    # Normalize Phi so g_tt matches Schwarzschild exterior at surface.
    phi_surface_raw = float(prof["phi_raw"].iloc[-1])
    phi_surface_target = 0.5 * math.log(max(1.0 - C, 1e-12))
    shift = phi_surface_target - phi_surface_raw
    prof["Phi"] = prof["phi_raw"] + shift
    prof["gtt_minus"] = -np.exp(2.0 * prof["Phi"])

    summary = {
        "eps_c_MeVfm3": eps_c_MeVfm3,
        "p_c_MeVfm3": p_c_MeVfm3,
        "B_MeVfm3": params.B_MeVfm3,
        "Delta_MeV": params.Delta_MeV,
        "mu_ref_MeV": params.mu_ref_MeV,
        "epsilon_surface_MeVfm3": epsilon_surface_MeVfm3(params),
        "pair_pressure_MeVfm3": cfl_pair_pressure_MeVfm3(params),
        "M_msun": float(prof["m_msun"].iloc[-1]),
        "R_km": float(prof["r_km"].iloc[-1]),
        "compactness_2M_R": C,
        "max_compactness_inside": float(prof["compactness_2m_r"].max()),
        "surface_f_metric": float(prof["f_metric"].iloc[-1]),
        "n_steps": len(prof),
        "status": "ok" if C < 1.0 else "static_caution",
    }
    return {"profile": prof, "summary": summary}


def savefig(fig_dir: Path, name: str) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(fig_dir / f"{name}.png", dpi=180)
    plt.savefig(fig_dir / f"{name}.pdf")
    plt.close()


def main() -> None:
    root = project_root()
    results = root / "results"
    fig_dir = root / "fig19_einstein_qcd_tov"
    results.mkdir(exist_ok=True)
    fig_dir.mkdir(exist_ok=True)

    cfg = TOVConfig()
    eos_grid = [
        EOSParams(B_MeVfm3=70, Delta_MeV=0),
        EOSParams(B_MeVfm3=80, Delta_MeV=50),
        EOSParams(B_MeVfm3=90, Delta_MeV=100),
    ]
    central_grid = np.linspace(350.0, 2200.0, 38)

    summaries = []
    representative_saved = False
    for eos in eos_grid:
        for epsc in central_grid:
            try:
                out = integrate_star(float(epsc), eos, cfg)
                summaries.append(out["summary"])
                # Save representative profile near fiducial central density.
                if (not representative_saved) and eos.B_MeVfm3 == 80 and eos.Delta_MeV == 50 and abs(epsc - 1200) < 40:
                    out["profile"].to_csv(results / "einstein_qcd_tov_representative_profile.csv", index=False)
                    representative_saved = True
            except Exception as exc:
                summaries.append({
                    "eps_c_MeVfm3": float(epsc),
                    "B_MeVfm3": eos.B_MeVfm3,
                    "Delta_MeV": eos.Delta_MeV,
                    "mu_ref_MeV": eos.mu_ref_MeV,
                    "epsilon_surface_MeVfm3": epsilon_surface_MeVfm3(eos),
                    "pair_pressure_MeVfm3": cfl_pair_pressure_MeVfm3(eos),
                    "M_msun": np.nan,
                    "R_km": np.nan,
                    "compactness_2M_R": np.nan,
                    "status": f"failed: {exc}",
                })

    df = pd.DataFrame(summaries)
    df.to_csv(results / "einstein_qcd_tov_mass_radius.csv", index=False)

    ok = df[df["status"].astype(str).str.startswith("ok")].copy()
    best = ok.sort_values("M_msun", ascending=False).groupby(["B_MeVfm3", "Delta_MeV"]).head(1)
    best.to_csv(results / "einstein_qcd_tov_max_mass_summary.csv", index=False)

    # EOS curve
    fid = EOSParams(B_MeVfm3=80, Delta_MeV=50)
    eps = np.linspace(epsilon_surface_MeVfm3(fid), 2200, 300)
    p = pressure_from_epsilon_MeVfm3(eps, fid)
    plt.figure(figsize=(7,5))
    plt.plot(eps, p, label=r"QCD-inspired linear CFL/bag EOS")
    plt.axhline(0, linestyle="--")
    plt.xlabel(r"$\epsilon$ [MeV fm$^{-3}$]")
    plt.ylabel(r"$p$ [MeV fm$^{-3}$]")
    plt.title("Einstein-QCD-inspired EOS")
    plt.legend()
    savefig(fig_dir, "fig19a_qcd_eos_pressure_energy")

    # Mass-radius curves
    plt.figure(figsize=(7,5))
    for (B, D), g in ok.groupby(["B_MeVfm3", "Delta_MeV"]):
        g = g.sort_values("eps_c_MeVfm3")
        plt.plot(g["R_km"], g["M_msun"], marker="o", markersize=3, label=f"B={B:g}, Delta={D:g}")
    plt.xlabel(r"$R$ [km]")
    plt.ylabel(r"$M/M_\odot$")
    plt.title("Einstein + QCD-inspired EOS mass-radius curves")
    plt.legend()
    savefig(fig_dir, "fig19b_mass_radius_curves")

    # Compactness vs central density
    plt.figure(figsize=(7,5))
    for (B, D), g in ok.groupby(["B_MeVfm3", "Delta_MeV"]):
        g = g.sort_values("eps_c_MeVfm3")
        plt.plot(g["eps_c_MeVfm3"], g["compactness_2M_R"], marker="o", markersize=3, label=f"B={B:g}, Delta={D:g}")
    plt.axhline(1.0, linestyle="--")
    plt.xlabel(r"$\epsilon_c$ [MeV fm$^{-3}$]")
    plt.ylabel(r"$2M/R$")
    plt.title("Compactness from Einstein-QCD-inspired TOV solutions")
    plt.legend()
    savefig(fig_dir, "fig19c_compactness_curve")

    # Representative profile
    prof_path = results / "einstein_qcd_tov_representative_profile.csv"
    if prof_path.exists():
        prof = pd.read_csv(prof_path)
        plt.figure(figsize=(7,5))
        plt.plot(prof["r_km"], prof["epsilon_MeVfm3"]/prof["epsilon_MeVfm3"].iloc[0], label=r"$\epsilon/\epsilon_c$")
        plt.plot(prof["r_km"], prof["p_MeVfm3"]/max(prof["p_MeVfm3"].iloc[0], 1e-12), label=r"$p/p_c$")
        plt.plot(prof["r_km"], prof["m_msun"]/max(prof["m_msun"].iloc[-1], 1e-12), label=r"$m/M$")
        plt.xlabel(r"$r$ [km]")
        plt.ylabel("normalized profile")
        plt.title("Representative Einstein-QCD-inspired TOV profile")
        plt.legend()
        savefig(fig_dir, "fig19d_representative_profiles")

        plt.figure(figsize=(7,5))
        plt.plot(prof["r_km"], prof["f_metric"], label=r"$f(r)=1-2m/r$")
        plt.plot(prof["r_km"], -prof["gtt_minus"], label=r"$e^{2\Phi(r)}$")
        plt.axhline(0, linestyle="--")
        plt.xlabel(r"$r$ [km]")
        plt.ylabel("metric function")
        plt.title("Representative metric diagnostics")
        plt.legend()
        savefig(fig_dir, "fig19e_metric_diagnostics")

    report = f"""# Einstein-QCD-inspired TOV solver report

EOS models scanned: {len(eos_grid)}
Central densities scanned per EOS: {len(central_grid)}
Successful TOV solutions: {len(ok)}

Boundary:
This is Einstein + QCD-inspired EOS/TOV modelling, not a full first-principles
Einstein-QCD solution.

Best maximum-mass rows:
{best.to_string(index=False)}
"""
    (results / "einstein_qcd_tov_report.md").write_text(report, encoding="utf-8")

    print("Einstein-QCD-inspired TOV solver completed.")
    print(f"Results: {results}")
    print(f"Figures: {fig_dir}")


if __name__ == "__main__":
    main()
