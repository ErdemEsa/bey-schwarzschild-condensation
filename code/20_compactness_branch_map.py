#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
20_compactness_branch_map.py

BEY patch-v3:
Compactness branch map for the tapered regular-core family.

This module summarizes how the dimensionless compactness
C = 2M/R_match = A I(x_match)/x_match
varies across A, x_match, and taper sharpness q.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class BranchParams:
    A: float = 1.0
    x_match: float = 4.0
    q: int = 2
    xmax: float = 6.0
    ngrid: int = 8000


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def cumulative_trapezoid_np(y, x):
    dx = np.diff(x)
    out = np.zeros_like(y, dtype=float)
    out[1:] = np.cumsum(0.5 * (y[1:] + y[:-1]) * dx)
    return out


def taper(x, x_match, q):
    y = x/x_match
    s = np.zeros_like(x)
    inside = y < 1
    s[inside] = (1-y[inside]**2)**q
    return s


def compactness(A, x_match, q):
    x = np.linspace(1e-4, 6.0, 8000)
    rho = (1/(1+x**4))*taper(x, x_match, q)
    I = cumulative_trapezoid_np(x**2*rho, x)
    idx = int(np.argmin(np.abs(x-x_match)))
    return float(A*I[idx]/x[idx])


def classify(C):
    if C < 0.8:
        return "subcompact"
    if C < 0.98:
        return "compact_below_horizon"
    if C < 1.0:
        return "horizon_near_ECO_branch"
    return "horizon_interior_static_caution"


def savefig(fig_dir, name):
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(fig_dir/f"{name}.png", dpi=180)
    plt.savefig(fig_dir/f"{name}.pdf")
    plt.close()


def main():
    root=project_root()
    results=root/"results"
    fig_dir=root/"fig18_compactness_branch_map"
    results.mkdir(exist_ok=True)
    fig_dir.mkdir(exist_ok=True)

    A_values=np.linspace(0.2, 24, 120)
    x_values=[3.0,4.0,5.0]
    q_values=[2,4]
    rows=[]
    for q in q_values:
        for xm in x_values:
            for A in A_values:
                C=compactness(float(A), xm, q)
                rows.append({"A":float(A),"x_match":xm,"q":q,"compactness_2M_over_R":C,"branch":classify(C)})
    df=pd.DataFrame(rows)
    df.to_csv(results/"compactness_branch_map.csv", index=False)

    # Threshold A for C=1 approximately
    thr=[]
    for (xm,q), g in df.groupby(["x_match","q"]):
        g=g.sort_values("A")
        below=g[g["compactness_2M_over_R"]<1]
        above=g[g["compactness_2M_over_R"]>=1]
        if len(below)>0 and len(above)>0:
            A_thr=float(above.iloc[0]["A"])
        else:
            A_thr=np.nan
        thr.append({"x_match":xm,"q":q,"A_threshold_C_ge_1":A_thr})
    pd.DataFrame(thr).to_csv(results/"compactness_thresholds.csv", index=False)

    plt.figure(figsize=(7,5))
    for (xm,q), g in df.groupby(["x_match","q"]):
        g=g.sort_values("A")
        plt.plot(g["A"], g["compactness_2M_over_R"], label=f"x={xm:g}, q={q}")
    plt.axhline(0.8, linestyle="--")
    plt.axhline(0.98, linestyle="--")
    plt.axhline(1.0, linestyle="--")
    plt.xlabel(r"$A=8\pi\rho_cR_c^2$")
    plt.ylabel(r"$2M/R_{\rm match}$")
    plt.title("BEY compactness branch map")
    plt.legend()
    savefig(fig_dir, "fig18a_compactness_branch_map")

    plt.figure(figsize=(7,5))
    for q, g in df.groupby("q"):
        # count class by A bins not ideal; show histogram of branch counts
        counts=g["branch"].value_counts()
        plt.bar([f"{q}:{k}" for k in counts.index], counts.values)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("count")
    plt.title("BEY compactness branch counts")
    savefig(fig_dir, "fig18b_branch_counts")

    report="""# BEY compactness branch map

This module maps C=2M/R_match for the tapered regular-core family.
C<1 corresponds to a matching surface outside the Schwarzschild radius.
C close to 1 is interpreted as the conditional horizon-near ECO branch.
C>=1 is labeled as horizon-interior/static-coordinate caution because a static
single-function ansatz is not a complete dynamical black-hole interior model.
"""
    (results/"compactness_branch_map_report.md").write_text(report, encoding="utf-8")

    print("BEY compactness branch map completed.")
    print(f"Results: {results}")
    print(f"Figures: {fig_dir}")


if __name__=="__main__":
    main()
