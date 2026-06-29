#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
23_anisotropic_cfl_core_solver.py

BEY / Schwarzschild Condensation patch-v6:
Anisotropic Einstein + CFL effective-pressure TOV solver.

Purpose
-------
This module extends the isotropic QCD-inspired TOV calculation by allowing an
effective anisotropic pressure channel,

    p_t(r) = p_r(r) + sigma(r)

with

    sigma(r) = alpha * p_r(r) * r^2/(r^2 + R_a^2)

so anisotropy vanishes at the centre and grows smoothly outward.

The anisotropic TOV equation is

    dp_r/dr = - (epsilon+p_r)(m+4*pi*r^3 p_r)/(r(r-2m)) + 2(p_t-p_r)/r

Boundary
--------
This is an effective anisotropic compact-core model. It is not a full
relativistic QCD transport calculation and not a nonlinear stability theorem.
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
class AnisoParams:
    alpha: float = 0.1
    R_aniso_km: float = 8.0


@dataclass(frozen=True)
class SolverConfig:
    r_max_km: float = 40.0
    dr_m: float = 25.0
    p_surface_frac: float = 1e-8


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def mevfm3_to_geo(x):
    return G * (np.asarray(x) * MEVFM3_TO_JM3) / c**4


def geo_to_mevfm3(x_geo):
    return np.asarray(x_geo) * c**4 / G / MEVFM3_TO_JM3


def mass_m_to_msun(m_geo_m):
    return np.asarray(m_geo_m) * c**2 / G / M_sun


def cfl_pair_pressure_MeVfm3(params: EOSParams) -> float:
    p_mev4 = 3.0 * params.Delta_MeV**2 * params.mu_ref_MeV**2 / (math.pi**2)
    return p_mev4 / (HBARC_MEV_FM**3)


def epsilon_surface_MeVfm3(params: EOSParams) -> float:
    return max(4.0 * params.B_MeVfm3 - 3.0 * cfl_pair_pressure_MeVfm3(params), params.epsilon_floor_MeVfm3)


def pressure_from_epsilon_MeVfm3(eps, params: EOSParams):
    return params.cs2 * (np.asarray(eps) - epsilon_surface_MeVfm3(params))


def epsilon_from_pressure_MeVfm3(p, params: EOSParams):
    return epsilon_surface_MeVfm3(params) + np.asarray(p) / params.cs2


def sigma_geo(r: float, p_r_geo: float, aniso: AnisoParams) -> float:
    Ra = aniso.R_aniso_km * 1000.0
    g = r**2 / (r**2 + Ra**2)
    return aniso.alpha * p_r_geo * g


def rhs(r: float, m: float, p_r: float, phi: float, eos: EOSParams, aniso: AnisoParams):
    if p_r <= 0:
        return 0.0, 0.0, 0.0
    eps = mevfm3_to_geo(epsilon_from_pressure_MeVfm3(geo_to_mevfm3(p_r), eos))
    sig = sigma_geo(r, p_r, aniso)
    p_t = p_r + sig
    denom = max(r * (r - 2.0*m), 1e-30)
    dm = 4.0 * math.pi * r**2 * eps
    grav = - (eps + p_r) * (m + 4.0 * math.pi * r**3 * p_r) / denom
    anisoterm = 2.0 * sig / max(r, 1e-30)
    dp = grav + anisoterm
    dphi = (m + 4.0 * math.pi * r**3 * p_r) / denom
    return dm, dp, dphi


def rk4_step(r, m, p, phi, h, eos, aniso):
    def f(rr, mm, pp, ph):
        return np.array(rhs(rr, mm, pp, ph, eos, aniso), dtype=float)
    y = np.array([m, p, phi], dtype=float)
    k1 = f(r, *y)
    k2 = f(r+0.5*h, *(y+0.5*h*k1))
    k3 = f(r+0.5*h, *(y+0.5*h*k2))
    k4 = f(r+h, *(y+h*k3))
    yn = y + (h/6.0)*(k1+2*k2+2*k3+k4)
    return float(yn[0]), float(yn[1]), float(yn[2])


def integrate_aniso(eps_c_MeVfm3: float, eos: EOSParams, aniso: AnisoParams, cfg: SolverConfig):
    p_c = pressure_from_epsilon_MeVfm3(eps_c_MeVfm3, eos)
    if p_c <= 0:
        raise ValueError("central density below surface density")

    dr = cfg.dr_m
    r = dr
    eps_c_geo = mevfm3_to_geo(eps_c_MeVfm3)
    p_geo = mevfm3_to_geo(p_c)
    m = 4.0 * math.pi * r**3 * eps_c_geo / 3.0
    phi = 0.0
    p_floor = max(p_geo * cfg.p_surface_frac, mevfm3_to_geo(1e-8))
    r_max = cfg.r_max_km * 1000.0

    rows = []
    while r < r_max and p_geo > p_floor and r > 2.0*m:
        eps_geo = mevfm3_to_geo(epsilon_from_pressure_MeVfm3(geo_to_mevfm3(p_geo), eos))
        sig = sigma_geo(r, p_geo, aniso)
        pt = p_geo + sig
        f_metric = 1.0 - 2.0*m/r

        # Energy-condition diagnostics in geometric units
        nec_r = eps_geo + p_geo
        nec_t = eps_geo + pt
        dec_r = eps_geo - abs(p_geo)
        dec_t = eps_geo - abs(pt)
        sec = eps_geo + p_geo + 2*pt

        rows.append({
            "r_km": r/1000.0,
            "m_msun": float(mass_m_to_msun(m)),
            "epsilon_MeVfm3": float(geo_to_mevfm3(eps_geo)),
            "p_r_MeVfm3": float(geo_to_mevfm3(p_geo)),
            "p_t_MeVfm3": float(geo_to_mevfm3(pt)),
            "sigma_MeVfm3": float(geo_to_mevfm3(sig)),
            "anisotropy_ratio_sigma_pr": float(sig/max(p_geo, 1e-30)),
            "f_metric": f_metric,
            "compactness_2m_r": 2*m/r,
            "NEC_r": float(nec_r),
            "NEC_t": float(nec_t),
            "SEC": float(sec),
            "DEC_r": float(dec_r),
            "DEC_t": float(dec_t),
            "phi_raw": phi,
        })

        m, p_geo, phi = rk4_step(r, m, p_geo, phi, dr, eos, aniso)
        r += dr
        if not np.isfinite(m) or not np.isfinite(p_geo) or not np.isfinite(phi):
            break

    prof = pd.DataFrame(rows)
    if prof.empty:
        raise RuntimeError("empty anisotropic profile")

    R_m = float(prof["r_km"].iloc[-1]*1000)
    M_m = float(prof["m_msun"].iloc[-1] * G * M_sun / c**2)
    C = 2*M_m/R_m

    phi_surface_raw = float(prof["phi_raw"].iloc[-1])
    phi_surface_target = 0.5*math.log(max(1-C, 1e-12))
    prof["Phi"] = prof["phi_raw"] + (phi_surface_target - phi_surface_raw)
    prof["gtt_minus"] = -np.exp(2*prof["Phi"])

    summary = {
        "eps_c_MeVfm3": eps_c_MeVfm3,
        "B_MeVfm3": eos.B_MeVfm3,
        "Delta_MeV": eos.Delta_MeV,
        "epsilon_surface_MeVfm3": epsilon_surface_MeVfm3(eos),
        "alpha": aniso.alpha,
        "R_aniso_km": aniso.R_aniso_km,
        "M_msun": float(prof["m_msun"].iloc[-1]),
        "R_km": float(prof["r_km"].iloc[-1]),
        "compactness_2M_R": C,
        "max_sigma_over_pr": float(prof["anisotropy_ratio_sigma_pr"].max()),
        "NEC_pass": bool((prof["NEC_r"] >= -1e-18).all() and (prof["NEC_t"] >= -1e-18).all()),
        "SEC_pass": bool((prof["SEC"] >= -1e-18).all()),
        "DEC_pass": bool((prof["DEC_r"] >= -1e-18).all() and (prof["DEC_t"] >= -1e-18).all()),
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


def main():
    root = project_root()
    results = root / "results"
    fig_dir = root / "fig20_anisotropic_cfl_core"
    results.mkdir(exist_ok=True)
    fig_dir.mkdir(exist_ok=True)

    eos = EOSParams(B_MeVfm3=80, Delta_MeV=50)
    cfg = SolverConfig()
    central_grid = np.linspace(400, 2200, 37)
    alpha_grid = [-0.20, -0.10, 0.0, 0.10, 0.20, 0.30]

    summaries = []
    rep_saved = False
    for alpha in alpha_grid:
        aniso = AnisoParams(alpha=float(alpha), R_aniso_km=8.0)
        for epsc in central_grid:
            try:
                out = integrate_aniso(float(epsc), eos, aniso, cfg)
                summaries.append(out["summary"])
                if (not rep_saved) and abs(alpha-0.2) < 1e-12 and abs(epsc-1200) < 40:
                    out["profile"].to_csv(results / "anisotropic_cfl_representative_profile.csv", index=False)
                    rep_saved = True
            except Exception as exc:
                summaries.append({
                    "eps_c_MeVfm3": float(epsc),
                    "B_MeVfm3": eos.B_MeVfm3,
                    "Delta_MeV": eos.Delta_MeV,
                    "alpha": float(alpha),
                    "M_msun": np.nan,
                    "R_km": np.nan,
                    "compactness_2M_R": np.nan,
                    "NEC_pass": False,
                    "SEC_pass": False,
                    "DEC_pass": False,
                    "status": f"failed: {exc}",
                })

    df = pd.DataFrame(summaries)
    df.to_csv(results / "anisotropic_cfl_tov_scan.csv", index=False)

    ok = df[df["status"].astype(str).str.startswith("ok")].copy()
    best = ok.sort_values("M_msun", ascending=False).groupby("alpha").head(1)
    best.to_csv(results / "anisotropic_cfl_max_mass_by_alpha.csv", index=False)

    # Plots
    plt.figure(figsize=(7,5))
    for alpha, g in ok.groupby("alpha"):
        g = g.sort_values("eps_c_MeVfm3")
        plt.plot(g["R_km"], g["M_msun"], marker="o", markersize=3, label=f"alpha={alpha:g}")
    plt.xlabel(r"$R$ [km]")
    plt.ylabel(r"$M/M_\odot$")
    plt.title("Anisotropic Einstein-CFL effective-core mass-radius curves")
    plt.legend()
    savefig(fig_dir, "fig20a_anisotropic_mass_radius")

    plt.figure(figsize=(7,5))
    for alpha, g in ok.groupby("alpha"):
        g = g.sort_values("eps_c_MeVfm3")
        plt.plot(g["eps_c_MeVfm3"], g["compactness_2M_R"], marker="o", markersize=3, label=f"alpha={alpha:g}")
    plt.axhline(1, linestyle="--")
    plt.xlabel(r"$\epsilon_c$ [MeV fm$^{-3}$]")
    plt.ylabel(r"$2M/R$")
    plt.title("Anisotropic compactness curves")
    plt.legend()
    savefig(fig_dir, "fig20b_anisotropic_compactness")

    plt.figure(figsize=(7,5))
    plt.plot(best["alpha"], best["M_msun"], marker="o")
    plt.xlabel(r"anisotropy strength $\alpha$")
    plt.ylabel(r"maximum $M/M_\odot$ in scan")
    plt.title("Maximum mass versus anisotropy strength")
    savefig(fig_dir, "fig20c_max_mass_vs_alpha")

    prof_path = results / "anisotropic_cfl_representative_profile.csv"
    if prof_path.exists():
        prof = pd.read_csv(prof_path)
        plt.figure(figsize=(7,5))
        plt.plot(prof["r_km"], prof["p_r_MeVfm3"], label=r"$p_r$")
        plt.plot(prof["r_km"], prof["p_t_MeVfm3"], label=r"$p_t$")
        plt.plot(prof["r_km"], prof["sigma_MeVfm3"], label=r"$p_t-p_r$")
        plt.axhline(0, linestyle="--")
        plt.xlabel(r"$r$ [km]")
        plt.ylabel(r"pressure [MeV fm$^{-3}$]")
        plt.title("Representative anisotropic CFL pressure profiles")
        plt.legend()
        savefig(fig_dir, "fig20d_pressure_anisotropy_profile")

        plt.figure(figsize=(7,5))
        plt.plot(prof["r_km"], prof["epsilon_MeVfm3"]/prof["epsilon_MeVfm3"].iloc[0], label=r"$\epsilon/\epsilon_c$")
        plt.plot(prof["r_km"], prof["m_msun"]/max(prof["m_msun"].iloc[-1], 1e-12), label=r"$m/M$")
        plt.plot(prof["r_km"], prof["compactness_2m_r"], label=r"$2m/r$")
        plt.xlabel(r"$r$ [km]")
        plt.ylabel("normalized profile")
        plt.title("Representative anisotropic effective-core profile")
        plt.legend()
        savefig(fig_dir, "fig20e_anisotropic_core_profile")

        plt.figure(figsize=(7,5))
        plt.plot(prof["r_km"], prof["DEC_r"], label=r"DEC$_r$")
        plt.plot(prof["r_km"], prof["DEC_t"], label=r"DEC$_t$")
        plt.plot(prof["r_km"], prof["SEC"], label="SEC")
        plt.axhline(0, linestyle="--")
        plt.xlabel(r"$r$ [km]")
        plt.ylabel("geometric condition value")
        plt.title("Representative anisotropic energy-condition diagnostics")
        plt.legend()
        savefig(fig_dir, "fig20f_anisotropic_energy_conditions")

    report = f"""# Anisotropic Einstein-CFL effective-core solver report

EOS: B={eos.B_MeVfm3} MeV/fm^3, Delta={eos.Delta_MeV} MeV
Alpha values scanned: {alpha_grid}
Central densities per alpha: {len(central_grid)}
Successful anisotropic solutions: {len(ok)}

Boundary:
This is an anisotropic Einstein + CFL-inspired effective-pressure model, not a
full relativistic QCD transport or nonlinear stability calculation.

Best rows by alpha:
{best.to_string(index=False)}
"""
    (results / "anisotropic_cfl_tov_report.md").write_text(report, encoding="utf-8")
    print("Anisotropic Einstein-CFL effective-core solver completed.")
    print(f"Results: {results}")
    print(f"Figures: {fig_dir}")


if __name__ == "__main__":
    main()
