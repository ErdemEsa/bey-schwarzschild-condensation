#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
19_curvature_regular_core.py

BEY / Schwarzschild Condensation patch-v3:
Curvature regularity and compactness diagnostics.

Purpose
-------
Patch-v1 established energy-condition diagnostics for a regular-core ansatz.
Patch-v2 introduced a tapered branch that suppresses density and pressure at the
matching surface.

Patch-v3 adds curvature diagnostics for the same effective metric family,

    ds^2 = -f(x) dt^2 + dx^2/f(x) + x^2 dOmega^2

in dimensionless units, with

    f(x) = 1 - A I(x)/x.

For the single-function metric, the curvature indicators are

    R(x) = -f'' - 4 f'/x + 2(1-f)/x^2

    K(x) = (f'')^2 + 4(f'/x)^2 + 4((1-f)/x^2)^2

These reproduce R=0 for Schwarzschild and finite constants for de Sitter-like
regular cores. Here they are used as regularity diagnostics, not as a full
Einstein-QCD solution.

Boundary
--------
This is a toy-model curvature/compactness diagnostic. It is not a complete
Darmois-Israel junction calculation and not a proof of singularity resolution
in quantum gravity.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class CurvatureParams:
    A: float = 1.0
    x_match: float = 4.0
    q: int = 2
    xmax: float = 6.0
    ngrid: int = 8000
    tapered: bool = True


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def cumulative_trapezoid_np(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    dx = np.diff(x)
    out = np.zeros_like(y, dtype=float)
    out[1:] = np.cumsum(0.5 * (y[1:] + y[:-1]) * dx)
    return out


def taper(x: np.ndarray, x_match: float, q: int) -> np.ndarray:
    y = x / x_match
    s = np.zeros_like(x, dtype=float)
    inside = y < 1.0
    s[inside] = (1.0 - y[inside] ** 2) ** q
    return s


def density_profile(params: CurvatureParams) -> dict[str, np.ndarray]:
    x = np.linspace(1e-4, params.xmax, params.ngrid)
    base = 1.0 / (1.0 + x**4)
    S = taper(x, params.x_match, params.q) if params.tapered else np.ones_like(x)
    rho = base * S
    I = cumulative_trapezoid_np(x**2 * rho, x)
    f = 1.0 - params.A * np.divide(I, x, out=np.zeros_like(I), where=x != 0)
    return {"x": x, "rho": rho, "I": I, "f": f, "base": base, "taper": S}


def curvature_indicators(params: CurvatureParams) -> dict[str, np.ndarray]:
    p = density_profile(params)
    x = p["x"]
    f = p["f"]

    # Use numerical derivatives. Edges are excluded in summaries to avoid
    # derivative artifacts.
    fp = np.gradient(f, x)
    fpp = np.gradient(fp, x)

    R = -fpp - 4.0 * fp / x + 2.0 * (1.0 - f) / (x**2)
    K = fpp**2 + 4.0 * (fp / x)**2 + 4.0 * ((1.0 - f) / (x**2))**2

    p.update({"fp": fp, "fpp": fpp, "Ricci_R": R, "K_proxy": K})
    return p


def classify_compactness(C: float) -> str:
    if C < 0.8:
        return "subcompact"
    if C < 0.98:
        return "compact_below_horizon"
    if C < 1.0:
        return "horizon_near_ECO_branch"
    return "horizon_interior_static_caution"


def scan_one(params: CurvatureParams) -> dict[str, object]:
    p = curvature_indicators(params)
    x = p["x"]
    idx = int(np.argmin(np.abs(x - params.x_match)))
    inside = x <= params.x_match

    # Drop the first few grid points and the exact taper boundary derivative
    # neighborhood to avoid numerical derivative endpoint artifacts.
    safe = inside & (x > 0.02) & (x < params.x_match - 0.02)
    if safe.sum() < 10:
        safe = inside & (x > 0.02)

    C = float(params.A * p["I"][idx] / x[idx])
    f_match = float(p["f"][idx])
    min_f = float(np.nanmin(p["f"][inside]))
    zero_crossings = int(np.sum(np.diff(np.signbit(p["f"][inside])) != 0))

    R = p["Ricci_R"][safe]
    K = p["K_proxy"][safe]

    finite_curvature = bool(np.all(np.isfinite(R)) and np.all(np.isfinite(K)))
    regular_center_proxy = bool(
        np.isfinite(p["rho"][0])
        and np.isfinite(p["f"][0])
        and np.isfinite(p["Ricci_R"][10])
        and np.isfinite(p["K_proxy"][10])
    )

    return {
        "profile": "tapered" if params.tapered else "untapered",
        "A": params.A,
        "x_match": params.x_match,
        "q": params.q,
        "compactness_2M_over_R": C,
        "compactness_class": classify_compactness(C),
        "f_match": f_match,
        "min_f_inside": min_f,
        "zero_crossings_inside": zero_crossings,
        "finite_curvature": finite_curvature,
        "regular_center_proxy": regular_center_proxy,
        "Ricci_min": float(np.nanmin(R)),
        "Ricci_max": float(np.nanmax(R)),
        "Ricci_abs_max": float(np.nanmax(np.abs(R))),
        "K_min": float(np.nanmin(K)),
        "K_max": float(np.nanmax(K)),
        "K_abs_max": float(np.nanmax(np.abs(K))),
        "rho_center": float(p["rho"][0]),
        "rho_boundary": float(p["rho"][idx]),
    }


def savefig(fig_dir: Path, name: str) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(fig_dir / f"{name}.png", dpi=180)
    plt.savefig(fig_dir / f"{name}.pdf")
    plt.close()


def plot_reference(params: CurvatureParams, fig_dir: Path) -> None:
    p = curvature_indicators(params)
    x = p["x"]
    inside = x <= params.x_match
    safe = inside & (x > 0.02) & (x < params.x_match - 0.02)

    plt.figure(figsize=(7, 5))
    plt.plot(x[inside], p["rho"][inside], label=r"$\rho/\rho_c$")
    plt.plot(x[inside], p["f"][inside], label=r"$f(x)$")
    plt.axvline(params.x_match, linestyle="--", label=r"$x_{\rm match}$")
    plt.axhline(0.0, linestyle="--")
    plt.xlabel(r"$x=r/R_c$")
    plt.ylabel("dimensionless value")
    plt.title("BEY tapered core profile and metric")
    plt.legend()
    savefig(fig_dir, "fig17a_profile_metric")

    plt.figure(figsize=(7, 5))
    plt.plot(x[safe], p["Ricci_R"][safe], label=r"$R(x)$ proxy")
    plt.axhline(0.0, linestyle="--")
    plt.xlabel(r"$x=r/R_c$")
    plt.ylabel("Ricci scalar proxy")
    plt.title("BEY regular-core Ricci diagnostic")
    plt.legend()
    savefig(fig_dir, "fig17b_ricci_proxy")

    plt.figure(figsize=(7, 5))
    plt.plot(x[safe], p["K_proxy"][safe], label=r"$K(x)$ proxy")
    plt.xlabel(r"$x=r/R_c$")
    plt.ylabel("Kretschmann-like proxy")
    plt.title("BEY regular-core Kretschmann-like diagnostic")
    plt.legend()
    savefig(fig_dir, "fig17c_kretschmann_proxy")

    plt.figure(figsize=(7, 5))
    plt.semilogy(x[safe], np.abs(p["Ricci_R"][safe]) + 1e-16, label=r"$|R|$")
    plt.semilogy(x[safe], p["K_proxy"][safe] + 1e-16, label=r"$K$")
    plt.xlabel(r"$x=r/R_c$")
    plt.ylabel("log diagnostic value")
    plt.title("BEY curvature diagnostics, log scale")
    plt.legend()
    savefig(fig_dir, "fig17d_curvature_log")


def plot_scan(df: pd.DataFrame, fig_dir: Path) -> None:
    tapered = df[df["profile"] == "tapered"].copy()

    plt.figure(figsize=(7, 5))
    for xm, g in tapered.groupby("x_match"):
        g = g.sort_values("A")
        plt.plot(g["A"], g["compactness_2M_over_R"], marker="o", label=f"x_match={xm:g}")
    plt.axhline(1.0, linestyle="--")
    plt.xlabel(r"$A=8\pi\rho_cR_c^2$")
    plt.ylabel(r"$2M/R_{\rm match}$")
    plt.title("BEY compactness scan for tapered branch")
    plt.legend()
    savefig(fig_dir, "fig17e_compactness_scan")

    plt.figure(figsize=(7, 5))
    for xm, g in tapered.groupby("x_match"):
        g = g.sort_values("compactness_2M_over_R")
        plt.plot(g["compactness_2M_over_R"], g["K_abs_max"], marker="o", label=f"x_match={xm:g}")
    plt.axvline(1.0, linestyle="--")
    plt.xlabel(r"$2M/R_{\rm match}$")
    plt.ylabel(r"max $K$ proxy")
    plt.title("BEY curvature proxy vs compactness")
    plt.legend()
    savefig(fig_dir, "fig17f_curvature_vs_compactness")


def main() -> None:
    root = project_root()
    results = root / "results"
    fig_dir = root / "fig17_curvature_regular_core"
    results.mkdir(exist_ok=True)
    fig_dir.mkdir(exist_ok=True)

    ref = CurvatureParams(A=1.0, x_match=4.0, q=2, tapered=True)
    plot_reference(ref, fig_dir)

    A_values = [0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 16.0, 20.0]
    x_match_values = [3.0, 4.0, 5.0]
    q_values = [2, 4]

    rows = []
    for tapered in [False, True]:
        for A in A_values:
            for xm in x_match_values:
                for q in q_values:
                    rows.append(scan_one(CurvatureParams(A=A, x_match=xm, q=q, tapered=tapered)))

    df = pd.DataFrame(rows)
    df.to_csv(results / "curvature_regular_core_scan.csv", index=False)

    summary = {
        "n_parameter_sets": len(df),
        "finite_curvature_count": int(df["finite_curvature"].sum()),
        "regular_center_proxy_count": int(df["regular_center_proxy"].sum()),
        "tapered_count": int((df["profile"] == "tapered").sum()),
        "untapered_count": int((df["profile"] == "untapered").sum()),
        "subcompact_count": int((df["compactness_class"] == "subcompact").sum()),
        "compact_below_horizon_count": int((df["compactness_class"] == "compact_below_horizon").sum()),
        "horizon_near_ECO_count": int((df["compactness_class"] == "horizon_near_ECO_branch").sum()),
        "horizon_interior_static_caution_count": int((df["compactness_class"] == "horizon_interior_static_caution").sum()),
    }
    pd.DataFrame([summary]).to_csv(results / "curvature_regular_core_summary.csv", index=False)

    # Recommended safe rows: tapered, finite, regular, C<1
    safe = df[
        (df["profile"] == "tapered")
        & (df["finite_curvature"])
        & (df["regular_center_proxy"])
        & (df["compactness_2M_over_R"] < 1.0)
    ].copy()
    safe.to_csv(results / "curvature_regular_core_safe_subhorizon.csv", index=False)

    plot_scan(df, fig_dir)

    report = f"""# BEY curvature regular-core diagnostics

Parameter sets scanned: {summary['n_parameter_sets']}

Finite curvature count:
{summary['finite_curvature_count']}

Regular-center proxy count:
{summary['regular_center_proxy_count']}

Compactness classes:
- subcompact: {summary['subcompact_count']}
- compact below horizon: {summary['compact_below_horizon_count']}
- horizon-near ECO branch: {summary['horizon_near_ECO_count']}
- horizon-interior static-coordinate caution: {summary['horizon_interior_static_caution_count']}

Interpretation:
The single-function regular-core metric produces finite Ricci and Kretschmann-like
curvature diagnostics over the scanned regularized domain. The compactness scan
separates subcompact, horizon-near ECO, and horizon-interior branches. The latter
requires careful interpretation because the static-coordinate ansatz is not a
complete dynamical black-hole interior construction.
"""
    (results / "curvature_regular_core_report.md").write_text(report, encoding="utf-8")

    print("BEY curvature regular-core diagnostics completed.")
    print(f"Results: {results}")
    print(f"Figures: {fig_dir}")


if __name__ == "__main__":
    main()
