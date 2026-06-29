#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
26_gl_free_energy_diagnostic.py

BEY patch-v7 post-processor for the fixed-background relativistic GL solution.
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


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
    profile_path = results / "gl_order_parameter_profile.csv"
    summary_path = results / "gl_order_parameter_summary.csv"
    if not profile_path.exists():
        raise FileNotFoundError("Run 25_relativistic_gl_order_parameter.py first.")

    prof = pd.read_csv(profile_path)
    summary = pd.read_csv(summary_path).iloc[0].to_dict() if summary_path.exists() else {}
    x = prof["x"].to_numpy(dtype=float)
    psi = prof["psi"].to_numpy(dtype=float)
    Wfree = prof["weighted_free_energy_density"].to_numpy(dtype=float)
    potential = prof["potential_density"].to_numpy(dtype=float)
    gradient = prof["gradient_density"].to_numpy(dtype=float)
    dx = float(x[1] - x[0])
    cumulative = np.concatenate([[0.0], np.cumsum(0.5 * (Wfree[:-1] + Wfree[1:]) * dx)])
    psi_norm = psi / max(float(psi.max()), 1e-12)
    core = psi_norm > 0.8
    interface = (psi_norm <= 0.8) & (psi_norm >= 0.2)
    tail = psi_norm < 0.2

    diag = {
        "background": summary.get("background", "unknown"),
        "total_free_energy_proxy": float(np.trapezoid(Wfree, x)),
        "gradient_energy_proxy": float(np.trapezoid(gradient, x)),
        "potential_energy_proxy": float(np.trapezoid(potential, x)),
        "psi_center": float(psi[0]),
        "psi_surface": float(psi[-1]),
        "psi_max": float(psi.max()),
        "x_at_half_max": float(x[np.argmin(np.abs(psi_norm - 0.5))]),
        "core_width_x_psi_gt_0p8": float((x[core].max() - x[core].min()) if core.any() else 0.0),
        "interface_width_x_0p2_to_0p8": float((x[interface].max() - x[interface].min()) if interface.any() else 0.0),
        "tail_width_x_psi_lt_0p2": float((x[tail].max() - x[tail].min()) if tail.any() else 0.0),
        "max_abs_residual": float(prof["gl_residual"].iloc[1:-1].abs().max()),
        "rms_residual": float(np.sqrt(np.mean(prof["gl_residual"].iloc[1:-1] ** 2))),
        "recommended_role": "supporting diagnostic / appendix candidate",
    }
    pd.DataFrame([diag]).to_csv(results / "gl_free_energy_diagnostics.csv", index=False)

    plt.figure(figsize=(7, 5))
    plt.plot(x, cumulative, label="cumulative free-energy proxy")
    plt.axhline(0, linestyle="--")
    plt.xlabel(r"$x=r/R$")
    plt.ylabel("cumulative proxy")
    plt.title("Cumulative GL free-energy diagnostic")
    plt.legend()
    savefig(fig_dir, "fig21e_cumulative_free_energy")

    plt.figure(figsize=(7, 5))
    plt.plot(x, psi_norm, label=r"$\psi/\psi_{\max}$")
    plt.fill_between(x, 0, 1, where=core, alpha=0.25, label="core")
    plt.fill_between(x, 0, 1, where=interface, alpha=0.25, label="interface")
    plt.fill_between(x, 0, 1, where=tail, alpha=0.20, label="tail")
    plt.xlabel(r"$x=r/R$")
    plt.ylabel("normalized amplitude")
    plt.title("GL condensate core-interface-tail decomposition")
    plt.legend()
    savefig(fig_dir, "fig21f_core_interface_tail")

    claim_note = """# Relativistic GL claim-safety note

Recommended wording:

As a relativistic order-parameter diagnostic, we solve a fixed-background GL
equation for a CFL-inspired condensate amplitude on the effective TOV geometry.

Avoid:

- We derive the CFL condensate from full QCD.
- We solve the fully backreacted Einstein-GL system.
- We prove the condensate is dynamically stable.

Best manuscript role:

Appendix or supplement-level diagnostic unless the final paper needs an explicit
relativistic complement to the nonrelativistic GP toy model.
"""
    (results / "gl_claim_safety_note.md").write_text(claim_note, encoding="utf-8")

    latex = r"""\subsection{Fixed-background relativistic GL diagnostic}
\label{sec:relativistic_gl_diagnostic}

As a relativistic order-parameter diagnostic, we solve a fixed-background GL
equation for a CFL-inspired condensate amplitude on the effective TOV geometry.
For a static spherical metric
\[
ds^2=-e^{2\Phi(r)}dt^2+\frac{dr^2}{f(r)}+r^2d\Omega^2,
\]
the radial order-parameter equation is
\[
\frac{1}{e^{\Phi}r^2}\frac{d}{dr}
\left(e^{\Phi}r^2 f(r)\frac{d\psi}{dr}\right)
=
a(r)\psi+b\psi^3 .
\]
The coefficient \(a(r)\) is chosen negative in the condensed region and positive
outside, so the solution interpolates between a finite condensate amplitude in
the core and a suppressed amplitude at the surface. This calculation is not a
microscopic derivation of CFL pairing from QCD and does not include metric
backreaction; it is a fixed-background relativistic consistency diagnostic.
"""
    (results / "gl_latex_snippet.tex").write_text(latex, encoding="utf-8")

    report = f"""# GL free-energy diagnostic report

Background: {diag['background']}
Total free-energy proxy: {diag['total_free_energy_proxy']:.6e}
Psi center: {diag['psi_center']:.6f}
Psi surface: {diag['psi_surface']:.6f}
x at half maximum: {diag['x_at_half_max']:.6f}
Core width psi>0.8: {diag['core_width_x_psi_gt_0p8']:.6f}
Interface width 0.2<psi<0.8: {diag['interface_width_x_0p2_to_0p8']:.6f}
Max residual: {diag['max_abs_residual']:.6e}
Recommended role: {diag['recommended_role']}

Interpretation:
The GL diagnostic supports a smooth relativistic order-parameter profile on the
fixed effective-core geometry. It should be used as a supporting diagnostic, not
as a first-principles CFL or fully backreacted Einstein-GL result.
"""
    (results / "gl_free_energy_report.md").write_text(report, encoding="utf-8")
    print("GL free-energy diagnostic completed.")
    print(report)


if __name__ == "__main__":
    main()
