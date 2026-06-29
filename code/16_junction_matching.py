#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
16_junction_matching.py

BEY / Schwarzschild Condensation matching diagnostics.

This script studies a simple regular-core interior matched to a Schwarzschild exterior
at x_match = R_match/R_c. It does not claim a full Israel-junction solution.
It produces diagnostic plots/tables for metric continuity and boundary pressure.

Boundary
--------
This is a matching diagnostic and thin-shell warning tool, not a complete proof
of a smooth interior-exterior solution.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class MatchParams:
    A: float = 1.0
    w: float = 0.3
    alpha: float = 0.2
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


def interior(params: MatchParams) -> dict[str, np.ndarray]:
    x = np.linspace(1e-5, params.xmax, params.ngrid)
    rho = 1.0 / (1.0 + x**4)
    I = cumulative_trapezoid_np(x**2 * rho, x)
    f_in = 1.0 - params.A * I / x
    g = x**2 / (1.0 + x**2)
    pr = params.w * rho
    pt = pr + params.alpha * rho * g
    return {"x": x, "rho": rho, "I": I, "f_in": f_in, "pr": pr, "pt": pt}


def match_diagnostics(params: MatchParams, x_match: float) -> dict[str, float]:
    p = interior(params)
    x = p["x"]
    idx = int(np.argmin(np.abs(x - x_match)))
    xm = float(x[idx])
    I_m = float(p["I"][idx])

    # In dimensionless x=r/Rc units:
    # f_in = 1 - A I/x.
    # If exterior mass is chosen as M=m(R_match), then 2M/R_match = A I(xm)/xm.
    compactness = params.A * I_m / xm
    f_out_match = 1.0 - compactness
    f_in_match = float(p["f_in"][idx])

    # Exterior Schwarzschild-like dimensionless f for x>=xm:
    f_out = 1.0 - compactness * xm / x

    # Derivative mismatch proxy. Exact Israel terms require a specific hypersurface
    # treatment; this proxy only signals whether a surface layer may be needed.
    df_in = np.gradient(p["f_in"], x)
    df_out = np.gradient(f_out, x)
    derivative_jump_proxy = float(df_out[idx] - df_in[idx])

    return {
        "A": params.A,
        "w": params.w,
        "alpha": params.alpha,
        "x_match": xm,
        "compactness_2M_over_R": compactness,
        "f_in_match": f_in_match,
        "f_out_match": f_out_match,
        "metric_jump_abs": abs(f_out_match - f_in_match),
        "df_jump_proxy": derivative_jump_proxy,
        "rho_boundary": float(p["rho"][idx]),
        "pr_boundary": float(p["pr"][idx]),
        "pt_boundary": float(p["pt"][idx]),
        "thin_shell_warning": bool(abs(float(p["pr"][idx])) > 1e-3 or abs(derivative_jump_proxy) > 1e-3),
    }


def savefig(fig_dir: Path, name: str) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(fig_dir / f"{name}.png", dpi=180)
    plt.savefig(fig_dir / f"{name}.pdf")
    plt.close()


def plot_matching(params: MatchParams, x_match: float, fig_dir: Path) -> None:
    p = interior(params)
    x = p["x"]
    idx = int(np.argmin(np.abs(x - x_match)))
    xm = float(x[idx])
    compactness = params.A * float(p["I"][idx]) / xm
    f_out = 1.0 - compactness * xm / x
    f_piece = np.where(x <= xm, p["f_in"], f_out)

    plt.figure(figsize=(7, 5))
    plt.plot(x, p["f_in"], label="interior f")
    plt.plot(x, f_out, label="exterior Schwarzschild f")
    plt.axvline(xm, linestyle="--", label=r"$x_{\rm match}$")
    plt.axhline(0.0, linestyle="--")
    plt.xlabel(r"$x=r/R_c$")
    plt.ylabel(r"$f(x)$")
    plt.title("BEY interior/exterior metric matching diagnostic")
    plt.legend()
    savefig(fig_dir, "fig14a_metric_matching")

    plt.figure(figsize=(7, 5))
    plt.plot(x, f_piece, label="piecewise matched f")
    plt.axvline(xm, linestyle="--", label=r"$x_{\rm match}$")
    plt.axhline(0.0, linestyle="--")
    plt.xlabel(r"$x=r/R_c$")
    plt.ylabel(r"$f(x)$")
    plt.title("BEY piecewise regular-core + Schwarzschild exterior")
    plt.legend()
    savefig(fig_dir, "fig14b_piecewise_metric")

    plt.figure(figsize=(7, 5))
    plt.plot(x, p["rho"], label=r"$\rho/\rho_c$")
    plt.plot(x, p["pr"], label=r"$p_r/\rho_c$")
    plt.plot(x, p["pt"], label=r"$p_t/\rho_c$")
    plt.axvline(xm, linestyle="--", label=r"$x_{\rm match}$")
    plt.axhline(0.0, linestyle="--")
    plt.xlabel(r"$x=r/R_c$")
    plt.ylabel("dimensionless profile")
    plt.title("BEY boundary density and pressure diagnostic")
    plt.legend()
    savefig(fig_dir, "fig14c_boundary_pressure")

    # Echo-extension plot, kept separate from the horizon-preserving branch.
    eps = np.logspace(-40, -2, 400)
    dt_over_M = 4.0 * np.abs(np.log(eps))
    plt.figure(figsize=(7, 5))
    plt.semilogx(eps, dt_over_M, label=r"$\Delta t_{\rm echo}/M \simeq 4|\ln\epsilon|$")
    plt.xlabel(r"$\epsilon$ in $R_{\rm surf}=2M(1+\epsilon)$")
    plt.ylabel(r"$\Delta t_{\rm echo}/M$")
    plt.title("Conditional ECO-branch echo delay scaling")
    plt.legend()
    savefig(fig_dir, "fig14d_echo_delay_scaling")


def main() -> None:
    root = project_root()
    results_dir = root / "results"
    fig_dir = root / "fig14_junction_matching"
    results_dir.mkdir(exist_ok=True)
    fig_dir.mkdir(exist_ok=True)

    params = MatchParams(A=1.0, w=0.3, alpha=0.2)
    x_values = [2.0, 3.0, 4.0, 5.0, 6.0]
    rows = [match_diagnostics(params, x) for x in x_values]
    df = pd.DataFrame(rows)
    df.to_csv(results_dir / "junction_matching_scan.csv", index=False)

    plot_matching(params, x_match=4.0, fig_dir=fig_dir)

    report = """# BEY junction/matching diagnostics

This diagnostic matches a regular-core interior to a Schwarzschild-like exterior by choosing
the exterior mass as M=m(R_match). This guarantees first-level f-continuity in the simple
dimensionless setup, but does not by itself prove a smooth Darmois-Israel matching.

Important:
- If radial pressure at the boundary is nonzero, a transition layer or thin-shell
  interpretation may be needed.
- If the derivative jump proxy is nonzero, extrinsic-curvature matching should be
  treated carefully in a later dedicated calculation.
- Echo delay scaling is shown only for the conditional horizon-near ECO branch.
"""
    (results_dir / "junction_matching_report.md").write_text(report, encoding="utf-8")

    print("BEY junction/matching diagnostics completed.")
    print(f"Results: {results_dir}")
    print(f"Figures: {fig_dir}")


if __name__ == "__main__":
    main()
