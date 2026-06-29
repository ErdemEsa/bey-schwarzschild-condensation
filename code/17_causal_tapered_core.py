#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
17_causal_tapered_core.py

BEY / Schwarzschild Condensation patch-v2:
Causal tapered regular-core diagnostics.

Motivation
----------
Patch-v1 showed that a simple regular-core ansatz gives broad NEC/WEC/DEC
acceptance, but a nonzero boundary pressure generically suggests a transition
layer or thin shell.

Patch-v2 introduces a compact/tapered effective core profile with:
    rho(x_match) = 0
    p_r(x_match) = 0
    p_t(x_match) = 0

This reduces boundary-pressure matching tension while preserving a regular center.

Boundary
--------
This is a toy-model consistency diagnostic, not a full Einstein-QCD solution and
not a complete GR stability proof.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class TaperParams:
    A: float = 1.0
    w: float = 0.3
    alpha: float = 0.0
    x_match: float = 4.0
    q: int = 2
    xmax: float = 6.0
    ngrid: int = 5000


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def cumulative_trapezoid_np(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    dx = np.diff(x)
    avg = 0.5 * (y[1:] + y[:-1])
    out = np.zeros_like(y, dtype=float)
    out[1:] = np.cumsum(avg * dx)
    return out


def taper(x: np.ndarray, x_match: float, q: int) -> np.ndarray:
    y = x / x_match
    s = np.zeros_like(x, dtype=float)
    inside = y < 1.0
    s[inside] = (1.0 - y[inside] ** 2) ** q
    return s


def profile(params: TaperParams) -> dict[str, np.ndarray]:
    x = np.linspace(1e-5, params.xmax, params.ngrid)

    base = 1.0 / (1.0 + x**4)
    S = taper(x, params.x_match, params.q)
    rho = base * S

    I = cumulative_trapezoid_np(x**2 * rho, x)
    f = 1.0 - params.A * np.divide(I, x, out=np.zeros_like(I), where=x != 0)

    # Causal baseline pressure: p_r=w*rho. This yields dp_r/d_rho=w.
    pr = params.w * rho

    # Boundary-safe anisotropy:
    # vanishes at center through y^2, and at boundary through S.
    y = x / params.x_match
    h = np.zeros_like(x)
    inside = y < 1.0
    h[inside] = y[inside] ** 2 / (1.0 + y[inside] ** 2)
    pt = pr + params.alpha * rho * h

    anis = pt - pr

    nec_r = rho + pr
    nec_t = rho + pt
    wec_0 = rho
    sec = rho + pr + 2.0 * pt
    dec_r = rho - np.abs(pr)
    dec_t = rho - np.abs(pt)

    cr2 = np.full_like(x, params.w, dtype=float)
    drho = np.gradient(rho, x)
    dpt = np.gradient(pt, x)
    ct2 = np.divide(dpt, drho, out=np.full_like(dpt, np.nan), where=np.abs(drho) > 1e-11)

    idx_match = int(np.argmin(np.abs(x - params.x_match)))

    return {
        "x": x,
        "base": base,
        "taper": S,
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
        "idx_match": idx_match,
    }


def finite_minmax(arr: np.ndarray) -> tuple[float, float]:
    arr = np.asarray(arr, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan, np.nan
    return float(np.nanmin(arr)), float(np.nanmax(arr))


def passes_nonnegative(arr: np.ndarray, tol: float = -1e-9) -> bool:
    arr = np.asarray(arr, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return False
    return bool(np.nanmin(arr) >= tol)


def scan_one(params: TaperParams) -> dict[str, object]:
    p = profile(params)
    x = p["x"]
    im = p["idx_match"]
    inside = x <= params.x_match

    nec_pass = passes_nonnegative(p["NEC_r"][inside]) and passes_nonnegative(p["NEC_t"][inside])
    wec_pass = passes_nonnegative(p["WEC_0"][inside]) and nec_pass
    sec_pass = passes_nonnegative(p["SEC"][inside])
    dec_pass = passes_nonnegative(p["DEC_r"][inside]) and passes_nonnegative(p["DEC_t"][inside])

    ct_inside = p["c_t2"][inside]
    ct_inside = ct_inside[np.isfinite(ct_inside)]
    causality_pass = (
        ct_inside.size > 10
        and np.nanmin(p["c_r2"][inside]) >= -1e-8
        and np.nanmax(p["c_r2"][inside]) <= 1.0 + 1e-8
        and np.nanmin(ct_inside) >= -1e-6
        and np.nanmax(ct_inside) <= 1.0 + 1e-6
    )

    center = x < 0.05
    center_regular = (
        np.all(np.isfinite(p["rho"][center]))
        and np.all(np.isfinite(p["f"][center]))
        and abs(float(p["anisotropy"][0])) < 1e-8
        and abs(float(p["m_norm"][0])) < 1e-8
    )

    boundary_pressure_abs = max(abs(float(p["pr"][im])), abs(float(p["pt"][im])))
    boundary_density_abs = abs(float(p["rho"][im]))
    boundary_soft = boundary_pressure_abs < 1e-5 and boundary_density_abs < 1e-5

    strict_accept = bool(nec_pass and wec_pass and dec_pass and causality_pass and center_regular and boundary_soft)
    regularizing_accept = bool(nec_pass and wec_pass and dec_pass and center_regular and boundary_soft)

    ct_min, ct_max = finite_minmax(ct_inside)

    return {
        "A": params.A,
        "w": params.w,
        "alpha": params.alpha,
        "x_match": params.x_match,
        "q": params.q,
        "NEC_pass": nec_pass,
        "WEC_pass": wec_pass,
        "SEC_pass": sec_pass,
        "DEC_pass": dec_pass,
        "causality_pass": causality_pass,
        "center_regular": center_regular,
        "boundary_soft": boundary_soft,
        "strict_accept": strict_accept,
        "regularizing_accept": regularizing_accept,
        "boundary_density_abs": boundary_density_abs,
        "boundary_pressure_abs": boundary_pressure_abs,
        "min_NEC_r": float(np.nanmin(p["NEC_r"][inside])),
        "min_NEC_t": float(np.nanmin(p["NEC_t"][inside])),
        "min_SEC": float(np.nanmin(p["SEC"][inside])),
        "min_DEC_r": float(np.nanmin(p["DEC_r"][inside])),
        "min_DEC_t": float(np.nanmin(p["DEC_t"][inside])),
        "min_f": float(np.nanmin(p["f"][inside])),
        "max_f": float(np.nanmax(p["f"][inside])),
        "min_c_t2": ct_min,
        "max_c_t2": ct_max,
    }


def savefig(fig_dir: Path, name: str) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(fig_dir / f"{name}.png", dpi=180)
    plt.savefig(fig_dir / f"{name}.pdf")
    plt.close()


def plot_reference(params: TaperParams, fig_dir: Path) -> None:
    p = profile(params)
    x = p["x"]
    inside = x <= params.x_match

    plt.figure(figsize=(7,5))
    plt.plot(x, p["base"], label="untapered base")
    plt.plot(x, p["taper"], label="compact taper")
    plt.plot(x, p["rho"], label=r"tapered $\rho/\rho_c$")
    plt.axvline(params.x_match, linestyle="--", label=r"$x_{\rm match}$")
    plt.xlabel(r"$x=r/R_c$")
    plt.ylabel("dimensionless profile")
    plt.title("BEY causal tapered density profile")
    plt.legend()
    savefig(fig_dir, "fig15a_tapered_density")

    plt.figure(figsize=(7,5))
    plt.plot(x, p["m_norm"], label=r"$m(x)/m(x_{\max})$")
    plt.axvline(params.x_match, linestyle="--", label=r"$x_{\rm match}$")
    plt.xlabel(r"$x=r/R_c$")
    plt.ylabel("normalized mass")
    plt.title("BEY tapered-core mass profile")
    plt.legend()
    savefig(fig_dir, "fig15b_tapered_mass")

    plt.figure(figsize=(7,5))
    plt.plot(x, p["f"], label=r"$f(x)$")
    plt.axhline(0, linestyle="--")
    plt.axvline(params.x_match, linestyle="--", label=r"$x_{\rm match}$")
    plt.xlabel(r"$x=r/R_c$")
    plt.ylabel(r"$f(x)$")
    plt.title("BEY tapered-core metric function")
    plt.legend()
    savefig(fig_dir, "fig15c_tapered_metric")

    plt.figure(figsize=(7,5))
    plt.plot(x[inside], p["NEC_r"][inside], label=r"NEC$_r$")
    plt.plot(x[inside], p["NEC_t"][inside], label=r"NEC$_t$")
    plt.plot(x[inside], p["DEC_r"][inside], label=r"DEC$_r$")
    plt.plot(x[inside], p["DEC_t"][inside], label=r"DEC$_t$")
    plt.axhline(0, linestyle="--")
    plt.xlabel(r"$x=r/R_c$")
    plt.ylabel("condition value")
    plt.title("BEY tapered-core NEC/DEC diagnostics")
    plt.legend()
    savefig(fig_dir, "fig15d_tapered_nec_dec")

    plt.figure(figsize=(7,5))
    plt.plot(x[inside], p["SEC"][inside], label="SEC")
    plt.axhline(0, linestyle="--")
    plt.xlabel(r"$x=r/R_c$")
    plt.ylabel("SEC value")
    plt.title("BEY tapered-core SEC diagnostic")
    plt.legend()
    savefig(fig_dir, "fig15e_tapered_sec")

    finite = np.isfinite(p["c_t2"]) & inside
    plt.figure(figsize=(7,5))
    plt.plot(x[inside], p["c_r2"][inside], label=r"$c_r^2$")
    plt.plot(x[finite], p["c_t2"][finite], label=r"$c_t^2$")
    plt.axhline(0, linestyle="--")
    plt.axhline(1, linestyle="--")
    plt.xlabel(r"$x=r/R_c$")
    plt.ylabel("sound-speed proxy")
    plt.title("BEY tapered-core causality proxies")
    plt.legend()
    savefig(fig_dir, "fig15f_tapered_causality")

    plt.figure(figsize=(7,5))
    plt.plot(x, p["pr"], label=r"$p_r$")
    plt.plot(x, p["pt"], label=r"$p_t$")
    plt.plot(x, p["anisotropy"], label=r"$p_t-p_r$")
    plt.axhline(0, linestyle="--")
    plt.axvline(params.x_match, linestyle="--", label=r"$x_{\rm match}$")
    plt.xlabel(r"$x=r/R_c$")
    plt.ylabel("dimensionless pressure")
    plt.title("BEY tapered-core pressure and anisotropy")
    plt.legend()
    savefig(fig_dir, "fig15g_tapered_pressure_anisotropy")


def plot_scan(df: pd.DataFrame, fig_dir: Path) -> None:
    grouped = df.groupby(["w", "alpha", "q"], as_index=False).agg(
        strict=("strict_accept", "mean"),
        regularizing=("regularizing_accept", "mean"),
        causal=("causality_pass", "mean"),
    )
    labels = [f"w={r.w:.1f}, a={r.alpha:.1f}, q={int(r.q)}" for _, r in grouped.iterrows()]
    x = np.arange(len(grouped))

    plt.figure(figsize=(12,5))
    plt.bar(x, grouped["strict"])
    plt.xticks(x, labels, rotation=70, ha="right")
    plt.ylim(0, 1.05)
    plt.ylabel("fraction over A and x_match grid")
    plt.title("BEY tapered-core strict acceptance")
    savefig(fig_dir, "fig15h_tapered_strict_acceptance")

    plt.figure(figsize=(12,5))
    plt.bar(x, grouped["regularizing"])
    plt.xticks(x, labels, rotation=70, ha="right")
    plt.ylim(0, 1.05)
    plt.ylabel("fraction over A and x_match grid")
    plt.title("BEY tapered-core regularizing acceptance")
    savefig(fig_dir, "fig15i_tapered_regularizing_acceptance")


def main() -> None:
    root = project_root()
    results = root / "results"
    fig_dir = root / "fig15_causal_tapered_core"
    results.mkdir(exist_ok=True)
    fig_dir.mkdir(exist_ok=True)

    ref = TaperParams(A=1.0, w=0.3, alpha=0.0, x_match=4.0, q=2)
    plot_reference(ref, fig_dir)

    A_values = [0.5, 1.0, 1.5, 2.0]
    w_values = [0.0, 0.2, 0.3, 0.5]
    alpha_values = [-0.2, 0.0, 0.2]
    x_match_values = [3.0, 4.0, 5.0]
    q_values = [2, 4]

    rows = []
    for A in A_values:
        for w in w_values:
            for alpha in alpha_values:
                for xm in x_match_values:
                    for q in q_values:
                        rows.append(scan_one(TaperParams(A=A, w=w, alpha=alpha, x_match=xm, q=q)))

    df = pd.DataFrame(rows)
    df.to_csv(results / "causal_tapered_core_scan.csv", index=False)

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
        "boundary_soft_count": int(df["boundary_soft"].sum()),
    }
    pd.DataFrame([summary]).to_csv(results / "causal_tapered_core_summary.csv", index=False)

    accepted = df[df["strict_accept"]].copy()
    accepted.to_csv(results / "causal_tapered_core_strict_accepted.csv", index=False)

    plot_scan(df, fig_dir)

    report = f"""# BEY causal tapered-core diagnostics

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
- Boundary soft: {summary['boundary_soft_count']}

Interpretation:
The tapered profile enforces rho, p_r, and p_t to vanish at the matching boundary.
This reduces boundary-pressure tension relative to the untapered patch-v1 core.
"""
    (results / "causal_tapered_core_report.md").write_text(report, encoding="utf-8")

    print("BEY causal tapered-core diagnostics completed.")
    print(f"Results: {results}")
    print(f"Figures: {fig_dir}")


if __name__ == "__main__":
    main()
