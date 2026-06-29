#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
18_smooth_transition_matching.py

BEY patch-v2 smooth transition / matching comparison.

Compares untapered and tapered regular-core matching diagnostics.

Boundary
--------
This is a metric and boundary-profile diagnostic, not a complete Darmois-Israel
junction calculation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class SmoothParams:
    A: float = 1.0
    w: float = 0.3
    alpha: float = 0.0
    x_match: float = 4.0
    q: int = 2
    xmax: float = 6.0
    ngrid: int = 5000


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def cumulative_trapezoid_np(y, x):
    dx = np.diff(x)
    out = np.zeros_like(y, dtype=float)
    out[1:] = np.cumsum(0.5 * (y[1:] + y[:-1]) * dx)
    return out


def taper(x, x_match, q):
    y = x / x_match
    s = np.zeros_like(x)
    inside = y < 1
    s[inside] = (1 - y[inside]**2)**q
    return s


def make_profile(params: SmoothParams, tapered: bool):
    x = np.linspace(1e-5, params.xmax, params.ngrid)
    base = 1/(1+x**4)
    S = taper(x, params.x_match, params.q) if tapered else np.ones_like(x)
    rho = base*S
    I = cumulative_trapezoid_np(x**2*rho, x)
    f = 1 - params.A*np.divide(I, x, out=np.zeros_like(I), where=x!=0)
    pr = params.w*rho
    y = x/params.x_match
    h = np.zeros_like(x)
    inside = y < 1
    h[inside] = y[inside]**2/(1+y[inside]**2)
    pt = pr + params.alpha*rho*h
    return {"x":x, "rho":rho, "I":I, "f":f, "pr":pr, "pt":pt}


def diagnostic(params: SmoothParams, tapered: bool):
    p = make_profile(params, tapered)
    x=p["x"]
    idx=int(np.argmin(np.abs(x-params.x_match)))
    xm=float(x[idx])
    compactness=float(params.A*p["I"][idx]/xm)
    f_out=1-compactness*xm/x
    df_in=np.gradient(p["f"], x)
    df_out=np.gradient(f_out, x)
    return {
        "profile": "tapered" if tapered else "untapered",
        "A": params.A,
        "w": params.w,
        "alpha": params.alpha,
        "x_match": xm,
        "q": params.q,
        "compactness_2M_over_R": compactness,
        "f_in_match": float(p["f"][idx]),
        "f_out_match": float(f_out[idx]),
        "metric_jump_abs": abs(float(p["f"][idx])-float(f_out[idx])),
        "df_jump_proxy": float(df_out[idx]-df_in[idx]),
        "rho_boundary": float(p["rho"][idx]),
        "pr_boundary": float(p["pr"][idx]),
        "pt_boundary": float(p["pt"][idx]),
        "thin_shell_warning": bool(max(abs(float(p["pr"][idx])),abs(float(p["pt"][idx]))) > 1e-5 or abs(float(df_out[idx]-df_in[idx])) > 1e-3)
    }


def savefig(fig_dir, name):
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(fig_dir/f"{name}.png", dpi=180)
    plt.savefig(fig_dir/f"{name}.pdf")
    plt.close()


def plot_comparison(params: SmoothParams, fig_dir: Path):
    pu=make_profile(params, tapered=False)
    pt=make_profile(params, tapered=True)
    x=pu["x"]
    idx=int(np.argmin(np.abs(x-params.x_match)))
    xm=x[idx]

    plt.figure(figsize=(7,5))
    plt.plot(x, pu["rho"], label="untapered rho")
    plt.plot(x, pt["rho"], label="tapered rho")
    plt.axvline(xm, linestyle="--", label=r"$x_{\rm match}$")
    plt.xlabel(r"$x=r/R_c$")
    plt.ylabel(r"$\rho/\rho_c$")
    plt.title("BEY untapered vs tapered boundary density")
    plt.legend()
    savefig(fig_dir, "fig16a_density_boundary_comparison")

    plt.figure(figsize=(7,5))
    plt.plot(x, pu["pr"], label="untapered p_r")
    plt.plot(x, pt["pr"], label="tapered p_r")
    plt.axvline(xm, linestyle="--", label=r"$x_{\rm match}$")
    plt.axhline(0, linestyle="--")
    plt.xlabel(r"$x=r/R_c$")
    plt.ylabel(r"$p_r/\rho_c$")
    plt.title("BEY untapered vs tapered boundary pressure")
    plt.legend()
    savefig(fig_dir, "fig16b_pressure_boundary_comparison")

    plt.figure(figsize=(7,5))
    plt.plot(x, pu["f"], label="untapered f")
    plt.plot(x, pt["f"], label="tapered f")
    plt.axvline(xm, linestyle="--", label=r"$x_{\rm match}$")
    plt.axhline(0, linestyle="--")
    plt.xlabel(r"$x=r/R_c$")
    plt.ylabel(r"$f(x)$")
    plt.title("BEY untapered vs tapered metric function")
    plt.legend()
    savefig(fig_dir, "fig16c_metric_comparison")


def main():
    root=project_root()
    results=root/"results"
    fig_dir=root/"fig16_smooth_transition_matching"
    results.mkdir(exist_ok=True)
    fig_dir.mkdir(exist_ok=True)

    rows=[]
    for A in [0.5,1.0,1.5,2.0]:
        for xm in [3.0,4.0,5.0]:
            params=SmoothParams(A=A, w=0.3, alpha=0.0, x_match=xm, q=2)
            rows.append(diagnostic(params, tapered=False))
            rows.append(diagnostic(params, tapered=True))
    df=pd.DataFrame(rows)
    df.to_csv(results/"smooth_transition_matching_comparison.csv", index=False)

    plot_comparison(SmoothParams(A=1.0,w=0.3,alpha=0.0,x_match=4.0,q=2), fig_dir)

    report="""# BEY smooth-transition matching comparison

This comparison shows how tapering the effective core profile toward the matching
surface reduces boundary density and radial pressure. Metric continuity is still
enforced by choosing M=m(R_match), while derivative-jump proxies remain diagnostic
signals for a possible transition layer.

Safe interpretation:
The tapered branch is better suited for smooth interior-exterior matching than the
untapered branch, but a complete Darmois-Israel treatment remains a later step.
"""
    (results/"smooth_transition_matching_report.md").write_text(report, encoding="utf-8")
    print("BEY smooth transition matching diagnostics completed.")
    print(f"Results: {results}")
    print(f"Figures: {fig_dir}")


if __name__=="__main__":
    main()
