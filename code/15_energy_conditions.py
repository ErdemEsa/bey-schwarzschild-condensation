#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
15_energy_conditions.py

BEY / Schwarzschild Condensation toy-model energy-condition diagnostics.

Purpose
-------
This script tests a minimal regular-core ansatz for the BEY model:

    rho(x) = rho_c / (1 + x^4),       x = r / R_c
    m(x)   = 4*pi*rho_c*R_c^3 * int_0^x u^2/(1+u^4) du
    f(x)   = 1 - A I(x)/x,            A = 8*pi*rho_c*R_c^2

with anisotropic effective pressures:

    p_r(x) = w rho(x)
    p_t(x) = p_r(x) + alpha rho(x) x^2/(1+x^2)

It computes NEC, WEC, SEC, DEC, causality diagnostics, and parameter acceptance.

Boundary
--------
This is a phenomenological regular-core consistency diagnostic.
It is not a full Einstein-QCD solution and not a complete GR stability proof.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class BEYParams:
    A: float = 1.0
    w: float = 0.0
    alpha: float = 0.0
    xmax: float = 6.0
    ngrid: int = 4000


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def cumulative_trapezoid_np(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    dx = np.diff(x)
    avg = 0.5 * (y[1:] + y[:-1])
    out = np.zeros_like(y, dtype=float)
    out[1:] = np.cumsum(avg * dx)
    return out


def profile(params: BEYParams) -> dict[str, np.ndarray]:
    x = np.linspace(1e-5, params.xmax, params.ngrid)
    rho = 1.0 / (1.0 + x**4)
    integrand = x**2 * rho
    I = cumulative_trapezoid_np(integrand, x)
    f = 1.0 - params.A * I / x

    g_aniso = x**2 / (1.0 + x**2)
    pr = params.w * rho
    pt = pr + params.alpha * rho * g_aniso
    anis = pt - pr

    nec_r = rho + pr
    nec_t = rho + pt
    wec_0 = rho
    sec = rho + pr + 2.0 * pt
    dec_r = rho - np.abs(pr)
    dec_t = rho - np.abs(pt)

    # Causality proxies. For p_r=w*rho, dp_r/d_rho = w.
    cr2 = np.full_like(x, params.w, dtype=float)

    # For p_t, compute dp_t/d_rho along the radial profile.
    # Avoid endpoint artifacts and division by near-zero gradient.
    drho = np.gradient(rho, x)
    dpt = np.gradient(pt, x)
    ct2 = np.divide(dpt, drho, out=np.full_like(dpt, np.nan), where=np.abs(drho) > 1e-12)

    return {
        "x": x,
        "rho": rho,
        "I": I,
        "m_norm": I / I[-1] if I[-1] != 0 else I,
        "f": f,
        "pr": pr,
        "pt": pt,
        "anisotropy": anis,
        "NEC_r": nec_r,
        "NEC_t": nec_t,
        "WEC_0": wec_0,
        "SEC": sec,
        "DEC_r": dec_r,
        "DEC_t": dec_t,
        "c_r2": cr2,
        "c_t2": ct2,
    }


def passes_condition(arr: np.ndarray, tol: float = -1e-10) -> bool:
    arr = np.asarray(arr, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return False
    return bool(np.nanmin(arr) >= tol)


def scan_one(params: BEYParams) -> dict[str, object]:
    p = profile(params)
    x = p["x"]
    center_window = x < 0.05

    nec_pass = passes_condition(p["NEC_r"]) and passes_condition(p["NEC_t"])
    wec_pass = passes_condition(p["WEC_0"]) and nec_pass
    sec_pass = passes_condition(p["SEC"])
    dec_pass = passes_condition(p["DEC_r"]) and passes_condition(p["DEC_t"])

    cr2 = p["c_r2"]
    ct2 = p["c_t2"]
    finite_ct = ct2[np.isfinite(ct2)]
    causality_pass = (
        finite_ct.size > 0
        and np.nanmin(cr2) >= -1e-8
        and np.nanmax(cr2) <= 1.0 + 1e-8
        and np.nanmin(finite_ct) >= -1e-8
        and np.nanmax(finite_ct) <= 1.0 + 1e-8
    )

    center_regular = (
        np.all(np.isfinite(p["rho"][center_window]))
        and np.all(np.isfinite(p["f"][center_window]))
        and abs(p["anisotropy"][0]) < 1e-6
        and abs(p["m_norm"][0]) < 1e-6
    )

    strict_accept = bool(nec_pass and wec_pass and dec_pass and causality_pass and center_regular)
    regularizing_accept = bool(nec_pass and wec_pass and dec_pass and center_regular)

    return {
        "A": params.A,
        "w": params.w,
        "alpha": params.alpha,
        "NEC_pass": nec_pass,
        "WEC_pass": wec_pass,
        "SEC_pass": sec_pass,
        "DEC_pass": dec_pass,
        "causality_pass": causality_pass,
        "center_regular": center_regular,
        "strict_accept": strict_accept,
        "regularizing_accept": regularizing_accept,
        "min_NEC_r": float(np.nanmin(p["NEC_r"])),
        "min_NEC_t": float(np.nanmin(p["NEC_t"])),
        "min_SEC": float(np.nanmin(p["SEC"])),
        "min_DEC_r": float(np.nanmin(p["DEC_r"])),
        "min_DEC_t": float(np.nanmin(p["DEC_t"])),
        "min_f": float(np.nanmin(p["f"])),
        "max_f": float(np.nanmax(p["f"])),
        "min_c_r2": float(np.nanmin(p["c_r2"])),
        "max_c_r2": float(np.nanmax(p["c_r2"])),
        "min_c_t2": float(np.nanmin(finite_ct)) if finite_ct.size else np.nan,
        "max_c_t2": float(np.nanmax(finite_ct)) if finite_ct.size else np.nan,
    }


def make_latex_table(df: pd.DataFrame, outpath: Path) -> None:
    cols = [
        "A", "w", "alpha", "NEC_pass", "WEC_pass", "SEC_pass",
        "DEC_pass", "causality_pass", "center_regular", "strict_accept",
        "regularizing_accept",
    ]
    lines = []
    lines.append(r"\begin{tabular}{ccc|ccccccc}")
    lines.append(r"\hline")
    lines.append(r"$A$ & $w$ & $\alpha$ & NEC & WEC & SEC & DEC & causal & regular & accept \\")
    lines.append(r"\hline")
    for _, r in df[cols].iterrows():
        def yn(v): return r"\checkmark" if bool(v) else r"$\times$"
        lines.append(
            f"{r['A']:.2f} & {r['w']:.2f} & {r['alpha']:.2f} & "
            f"{yn(r['NEC_pass'])} & {yn(r['WEC_pass'])} & {yn(r['SEC_pass'])} & "
            f"{yn(r['DEC_pass'])} & {yn(r['causality_pass'])} & "
            f"{yn(r['center_regular'])} & {yn(r['strict_accept'])} \\\\"
        )
    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    outpath.write_text("\n".join(lines), encoding="utf-8")


def savefig(fig_dir: Path, name: str) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(fig_dir / f"{name}.png", dpi=180)
    plt.savefig(fig_dir / f"{name}.pdf")
    plt.close()


def plot_reference_model(params: BEYParams, fig_dir: Path) -> None:
    p = profile(params)
    x = p["x"]

    plt.figure(figsize=(7, 5))
    plt.plot(x, p["rho"], label=r"$\rho/\rho_c$")
    plt.plot(x, p["m_norm"], label=r"$m(x)/m(x_{\max})$")
    plt.xlabel(r"$x=r/R_c$")
    plt.ylabel("dimensionless value")
    plt.title("BEY regular-core density and mass profile")
    plt.legend()
    savefig(fig_dir, "fig13a_density_mass_profile")

    plt.figure(figsize=(7, 5))
    plt.plot(x, p["f"], label=r"$f(x)=1-AI(x)/x$")
    plt.axhline(0.0, linestyle="--")
    plt.xlabel(r"$x=r/R_c$")
    plt.ylabel(r"$f(x)$")
    plt.title("BEY interior metric function")
    plt.legend()
    savefig(fig_dir, "fig13b_metric_function")

    plt.figure(figsize=(7, 5))
    plt.plot(x, p["NEC_r"], label=r"NEC$_r=\rho+p_r$")
    plt.plot(x, p["NEC_t"], label=r"NEC$_t=\rho+p_t$")
    plt.axhline(0.0, linestyle="--")
    plt.xlabel(r"$x=r/R_c$")
    plt.ylabel("condition value")
    plt.title("BEY null energy-condition diagnostics")
    plt.legend()
    savefig(fig_dir, "fig13c_nec")

    plt.figure(figsize=(7, 5))
    plt.plot(x, p["SEC"], label=r"SEC=$\rho+p_r+2p_t$")
    plt.axhline(0.0, linestyle="--")
    plt.xlabel(r"$x=r/R_c$")
    plt.ylabel("condition value")
    plt.title("BEY strong energy-condition diagnostic")
    plt.legend()
    savefig(fig_dir, "fig13d_sec")

    plt.figure(figsize=(7, 5))
    plt.plot(x, p["DEC_r"], label=r"DEC$_r=\rho-|p_r|$")
    plt.plot(x, p["DEC_t"], label=r"DEC$_t=\rho-|p_t|$")
    plt.axhline(0.0, linestyle="--")
    plt.xlabel(r"$x=r/R_c$")
    plt.ylabel("condition value")
    plt.title("BEY dominant energy-condition diagnostics")
    plt.legend()
    savefig(fig_dir, "fig13e_dec")

    plt.figure(figsize=(7, 5))
    plt.plot(x, p["anisotropy"], label=r"$p_t-p_r$")
    plt.axhline(0.0, linestyle="--")
    plt.xlabel(r"$x=r/R_c$")
    plt.ylabel("anisotropy")
    plt.title("BEY anisotropy profile")
    plt.legend()
    savefig(fig_dir, "fig13f_anisotropy")

    finite = np.isfinite(p["c_t2"])
    plt.figure(figsize=(7, 5))
    plt.plot(x, p["c_r2"], label=r"$c_r^2$")
    plt.plot(x[finite], p["c_t2"][finite], label=r"$c_t^2$")
    plt.axhline(0.0, linestyle="--")
    plt.axhline(1.0, linestyle="--")
    plt.xlabel(r"$x=r/R_c$")
    plt.ylabel("sound-speed proxy")
    plt.title("BEY causality proxies")
    plt.legend()
    savefig(fig_dir, "fig13g_causality")


def plot_acceptance(df: pd.DataFrame, fig_dir: Path) -> None:
    grouped = df.groupby(["w", "alpha"], as_index=False).agg(
        strict_accept_fraction=("strict_accept", "mean"),
        regularizing_accept_fraction=("regularizing_accept", "mean"),
        sec_fraction=("SEC_pass", "mean"),
    )

    labels = [f"w={r.w:.1f}, a={r.alpha:.1f}" for _, r in grouped.iterrows()]
    xpos = np.arange(len(grouped))

    plt.figure(figsize=(10, 5))
    plt.bar(xpos, grouped["strict_accept_fraction"])
    plt.xticks(xpos, labels, rotation=60, ha="right")
    plt.ylim(0.0, 1.05)
    plt.ylabel("fraction over A-grid")
    plt.title("BEY strict acceptance over parameter scan")
    savefig(fig_dir, "fig13h_strict_acceptance")

    plt.figure(figsize=(10, 5))
    plt.bar(xpos, grouped["regularizing_accept_fraction"])
    plt.xticks(xpos, labels, rotation=60, ha="right")
    plt.ylim(0.0, 1.05)
    plt.ylabel("fraction over A-grid")
    plt.title("BEY regularizing-core acceptance over parameter scan")
    savefig(fig_dir, "fig13i_regularizing_acceptance")


def main() -> None:
    root = project_root()
    results_dir = root / "results"
    fig_dir = root / "fig13_energy_conditions"
    results_dir.mkdir(exist_ok=True)
    fig_dir.mkdir(exist_ok=True)

    # Reference model:
    # w=0.3, alpha=0.2 is a conservative energy-condition-friendly toy core.
    # More regularizing negative-pressure variants are explored in the scan.
    ref = BEYParams(A=1.0, w=0.3, alpha=0.2)
    plot_reference_model(ref, fig_dir)

    rows = []
    A_values = [0.5, 1.0, 1.5, 2.0]
    w_values = [-0.8, -0.5, -0.3, 0.0, 0.3]
    alpha_values = [-0.5, 0.0, 0.5]
    for A in A_values:
        for w in w_values:
            for alpha in alpha_values:
                rows.append(scan_one(BEYParams(A=A, w=w, alpha=alpha)))

    df = pd.DataFrame(rows)
    df.to_csv(results_dir / "energy_conditions_scan.csv", index=False)
    make_latex_table(df, results_dir / "table_energy_conditions.tex")
    plot_acceptance(df, fig_dir)

    summary = {
        "n_parameter_sets": len(df),
        "strict_accept_count": int(df["strict_accept"].sum()),
        "regularizing_accept_count": int(df["regularizing_accept"].sum()),
        "NEC_pass_count": int(df["NEC_pass"].sum()),
        "WEC_pass_count": int(df["WEC_pass"].sum()),
        "SEC_pass_count": int(df["SEC_pass"].sum()),
        "DEC_pass_count": int(df["DEC_pass"].sum()),
        "causality_pass_count": int(df["causality_pass"].sum()),
        "center_regular_count": int(df["center_regular"].sum()),
    }
    pd.DataFrame([summary]).to_csv(results_dir / "energy_conditions_summary.csv", index=False)

    report = f"""# BEY energy-condition diagnostics

Parameter sets scanned: {summary['n_parameter_sets']}

Strict accepted sets:
{summary['strict_accept_count']}

Regularizing-core accepted sets:
{summary['regularizing_accept_count']}

Condition counts:
- NEC pass: {summary['NEC_pass_count']}
- WEC pass: {summary['WEC_pass_count']}
- SEC pass: {summary['SEC_pass_count']}
- DEC pass: {summary['DEC_pass_count']}
- Causality pass: {summary['causality_pass_count']}
- Center regular: {summary['center_regular_count']}

Boundary:
This is a phenomenological regular-core consistency diagnostic, not a complete GR stability proof.
"""
    (results_dir / "energy_conditions_report.md").write_text(report, encoding="utf-8")

    print("BEY energy-condition diagnostics completed.")
    print(f"Results: {results_dir}")
    print(f"Figures: {fig_dir}")


if __name__ == "__main__":
    main()
