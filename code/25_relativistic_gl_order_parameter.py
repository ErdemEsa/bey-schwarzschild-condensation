#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
25_relativistic_gl_order_parameter.py

BEY patch-v7: fixed-background relativistic Ginzburg-Landau order-parameter diagnostic.

This is not a microscopic CFL derivation from QCD and not a backreacted
Einstein-GL solution. It solves a relativistic GL amplitude equation on a fixed
effective TOV/regular-core geometry.
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class GLParams:
    a0: float = 80.0
    b: float = 1.0
    x_transition: float = 0.72
    width: float = 0.055
    n_grid: int = 320
    max_iter: int = 1800
    dt: float = 0.002
    tol: float = 2.5e-8


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_background(root: Path, n_grid: int) -> pd.DataFrame:
    candidates = [
        root / "results" / "einstein_qcd_tov_representative_profile.csv",
        root / "results_diagnostics" / "patch_v6_einstein_cfl" / "einstein_qcd_tov_representative_profile.csv",
    ]
    for path in candidates:
        if path.exists():
            df = pd.read_csv(path)
            r = df["r_km"].to_numpy(dtype=float)
            R = float(r[-1])
            x_old = r / R
            x = np.linspace(max(x_old[0], 1e-5), 1.0, n_grid)
            f = np.interp(x, x_old, df["f_metric"].to_numpy(dtype=float))
            if "Phi" in df:
                Phi = np.interp(x, x_old, df["Phi"].to_numpy(dtype=float))
            elif "gtt_minus" in df:
                gtt_abs = np.maximum(-df["gtt_minus"].to_numpy(dtype=float), 1e-14)
                Phi = 0.5 * np.log(np.interp(x, x_old, gtt_abs))
            else:
                Phi = 0.5 * np.log(np.maximum(f, 1e-14))
            return pd.DataFrame({"x": x, "r_km": x * R, "f_metric": f, "Phi": Phi, "background": "v6_tov"})

    x = np.linspace(1e-5, 1.0, n_grid)
    R = 12.0
    C = 0.55
    f = 1.0 - C * x**2
    Phi = 0.5 * np.log(np.maximum(1.0 - C * x**2, 1e-14))
    return pd.DataFrame({"x": x, "r_km": x * R, "f_metric": f, "Phi": Phi, "background": "fallback_regular"})


def a_profile(x: np.ndarray, p: GLParams) -> np.ndarray:
    return p.a0 * np.tanh((x - p.x_transition) / p.width)


def build_operator(x: np.ndarray, f: np.ndarray, Phi: np.ndarray):
    n = len(x)
    dx = float(x[1] - x[0])
    W = np.exp(Phi) * np.maximum(x**2, 1e-12)
    A_mid = 0.5 * (W[:-1] * f[:-1] + W[1:] * f[1:])
    lower = np.zeros(n)
    diag = np.zeros(n)
    upper = np.zeros(n)
    for i in range(1, n - 1):
        Am = A_mid[i - 1]
        Ap = A_mid[i]
        Wi = W[i]
        lower[i] = Am / (Wi * dx**2)
        upper[i] = Ap / (Wi * dx**2)
        diag[i] = -(Am + Ap) / (Wi * dx**2)
    return lower, diag, upper, W


def tridiag(lower, diag, upper, rhs):
    n = len(rhs)
    a, b, c, d = lower.copy(), diag.copy(), upper.copy(), rhs.copy()
    for i in range(1, n):
        if abs(b[i - 1]) < 1e-30:
            b[i - 1] = 1e-30
        w = a[i] / b[i - 1]
        b[i] -= w * c[i - 1]
        d[i] -= w * d[i - 1]
    x = np.zeros(n)
    if abs(b[-1]) < 1e-30:
        b[-1] = 1e-30
    x[-1] = d[-1] / b[-1]
    for i in range(n - 2, -1, -1):
        if abs(b[i]) < 1e-30:
            b[i] = 1e-30
        x[i] = (d[i] - c[i] * x[i + 1]) / b[i]
    return x


def L_apply(psi, lower, diag, upper):
    y = diag * psi
    y[1:] += lower[1:] * psi[:-1]
    y[:-1] += upper[:-1] * psi[1:]
    y[0] = y[1]
    y[-1] = y[-2]
    return y


def solve_gl(bg: pd.DataFrame, p: GLParams):
    x = bg["x"].to_numpy(dtype=float)
    f = np.maximum(bg["f_metric"].to_numpy(dtype=float), 1e-8)
    Phi = bg["Phi"].to_numpy(dtype=float)
    a = a_profile(x, p)
    lower_L, diag_L, upper_L, W = build_operator(x, f, Phi)

    psi = np.sqrt(np.maximum(-a / p.b, 0.0))
    psi_center_bc = float(np.sqrt(max(-a[0] / p.b, 0.0)))
    psi[-1] = 0.0
    psi[0] = psi_center_bc
    converged = False
    last_delta = np.inf

    for it in range(p.max_iter):
        lower = -p.dt * lower_L
        diag = 1.0 - p.dt * diag_L + p.dt * a + p.dt * p.b * psi**2
        upper = -p.dt * upper_L
        rhs = psi.copy()
        diag[0], upper[0], lower[0], rhs[0] = 1.0, -1.0, 0.0, 0.0
        diag[-1], lower[-1], upper[-1], rhs[-1] = 1.0, 0.0, 0.0, 0.0
        new = tridiag(lower, diag, upper, rhs)
        new = np.maximum(new, 0.0)
        new[-1] = 0.0
        new[0] = new[1]
        last_delta = float(np.max(np.abs(new - psi)))
        psi = new
        if last_delta < p.tol:
            converged = True
            break

    dx = float(x[1] - x[0])
    Lpsi = L_apply(psi, lower_L, diag_L, upper_L)
    residual = Lpsi - a * psi - p.b * psi**3
    dpsi = np.gradient(psi, dx)
    grad = 0.5 * f * dpsi**2
    pot = 0.5 * a * psi**2 + 0.25 * p.b * psi**4
    weighted = W * (grad + pot)

    out = bg.copy()
    out["a_gl"] = a
    out["psi"] = psi
    out["dpsi_dx"] = dpsi
    out["gradient_density"] = grad
    out["potential_density"] = pot
    out["weighted_free_energy_density"] = weighted
    out["gl_residual"] = residual

    interface = (psi > 0.1) & (psi < 0.9)
    summary = {
        "background": str(out["background"].iloc[0]),
        "n_grid": len(x),
        "converged": bool(converged),
        "iterations": int(it + 1),
        "last_delta": last_delta,
        "max_abs_residual": float(np.max(np.abs(residual[1:-1]))),
        "rms_residual": float(np.sqrt(np.mean(residual[1:-1] ** 2))),
        "total_free_energy_proxy": float(np.trapezoid(weighted, x)),
        "gradient_energy_proxy": float(np.trapezoid(W * grad, x)),
        "potential_energy_proxy": float(np.trapezoid(W * pot, x)),
        "psi_center": float(psi[0]),
        "psi_surface": float(psi[-1]),
        "psi_max": float(psi.max()),
        "condensed_weighted_fraction_psi_gt_0p5": float(np.trapezoid((psi > 0.5).astype(float) * W, x) / np.trapezoid(W, x)),
        "interface_width_in_x": float((x[interface].max() - x[interface].min()) if interface.any() else 0.0),
        "x_transition": p.x_transition,
        "width": p.width,
        "a0": p.a0,
        "b": p.b,
    }
    return out, summary


def savefig(fig_dir: Path, name: str):
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(fig_dir / f"{name}.png", dpi=180)
    plt.savefig(fig_dir / f"{name}.pdf")
    plt.close()


def main():
    root = project_root()
    results = root / "results"
    fig_dir = root / "fig21_relativistic_gl"
    results.mkdir(exist_ok=True)
    fig_dir.mkdir(exist_ok=True)

    params = GLParams()
    bg = load_background(root, params.n_grid)
    prof, summary = solve_gl(bg, params)
    prof.to_csv(results / "gl_order_parameter_profile.csv", index=False)
    pd.DataFrame([summary]).to_csv(results / "gl_order_parameter_summary.csv", index=False)

    plt.figure(figsize=(7, 5))
    plt.plot(prof["x"], prof["psi"], label=r"$\psi(x)$")
    plt.plot(prof["x"], np.maximum(-prof["a_gl"], 0.0), linestyle="--", label=r"$\max[-a(x),0]$")
    plt.axvline(params.x_transition, linestyle="--", label="transition scale")
    plt.xlabel(r"$x=r/R$")
    plt.ylabel("dimensionless amplitude")
    plt.title("Fixed-background relativistic GL order parameter")
    plt.legend()
    savefig(fig_dir, "fig21a_order_parameter_profile")

    plt.figure(figsize=(7, 5))
    plt.plot(prof["x"], prof["psi"] / max(prof["psi"].max(), 1e-12), label=r"$\psi/\psi_{\max}$")
    plt.plot(prof["x"], prof["f_metric"], label=r"$f(r)$")
    plt.plot(prof["x"], np.exp(2 * prof["Phi"]), label=r"$e^{2\Phi}$")
    plt.xlabel(r"$x=r/R$")
    plt.ylabel("normalized / metric value")
    plt.title("Metric background and GL condensate support")
    plt.legend()
    savefig(fig_dir, "fig21b_metric_vs_condensate")

    plt.figure(figsize=(7, 5))
    plt.plot(prof["x"], prof["gradient_density"], label="gradient")
    plt.plot(prof["x"], prof["potential_density"], label="potential")
    plt.plot(prof["x"], prof["weighted_free_energy_density"], label="weighted total")
    plt.axhline(0, linestyle="--")
    plt.xlabel(r"$x=r/R$")
    plt.ylabel("dimensionless density")
    plt.title("GL free-energy density diagnostics")
    plt.legend()
    savefig(fig_dir, "fig21c_free_energy_density")

    plt.figure(figsize=(7, 5))
    plt.plot(prof["x"], np.abs(prof["gl_residual"]))
    plt.yscale("log")
    plt.xlabel(r"$x=r/R$")
    plt.ylabel(r"$|\mathcal{R}_{GL}|$")
    plt.title("GL equation residual")
    savefig(fig_dir, "fig21d_gl_residual")

    report = f"""# Relativistic GL order-parameter report

Background: {summary['background']}
Converged: {summary['converged']}
Iterations: {summary['iterations']}
Max residual: {summary['max_abs_residual']:.6e}
RMS residual: {summary['rms_residual']:.6e}
Total free-energy proxy: {summary['total_free_energy_proxy']:.6e}
Gradient energy proxy: {summary['gradient_energy_proxy']:.6e}
Potential energy proxy: {summary['potential_energy_proxy']:.6e}
Psi centre: {summary['psi_center']:.6f}
Psi surface: {summary['psi_surface']:.6f}
Condensed weighted fraction psi>0.5: {summary['condensed_weighted_fraction_psi_gt_0p5']:.6f}

Boundary:
This is a fixed-background relativistic GL order-parameter diagnostic, not a
first-principles CFL derivation and not a backreacted Einstein-GL solution.
"""
    (results / "gl_order_parameter_report.md").write_text(report, encoding="utf-8")
    print("Relativistic GL order-parameter solver completed.")
    print(report)


if __name__ == "__main__":
    main()
