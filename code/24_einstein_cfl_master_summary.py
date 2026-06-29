#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
24_einstein_cfl_master_summary.py

BEY patch-v6 master summary for isotropic and anisotropic Einstein+QCD/CFL
effective-core solutions.
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def read(path: Path):
    return pd.read_csv(path) if path.exists() else None


def main():
    root = project_root()
    results = root / "results"
    results.mkdir(exist_ok=True)

    iso = read(results / "einstein_qcd_tov_mass_radius.csv")
    iso_best = read(results / "einstein_qcd_tov_max_mass_summary.csv")
    aniso = read(results / "anisotropic_cfl_tov_scan.csv")
    aniso_best = read(results / "anisotropic_cfl_max_mass_by_alpha.csv")

    rows = []
    if iso is not None:
        ok = iso[iso["status"].astype(str).str.startswith("ok")]
        rows.append({
            "module": "Einstein + QCD-inspired EOS / isotropic TOV",
            "solutions": len(ok),
            "max_mass_Msun": float(ok["M_msun"].max()) if len(ok) else np.nan,
            "max_compactness": float(ok["compactness_2M_R"].max()) if len(ok) else np.nan,
            "role_in_BEY": "supports Einstein-system effective-core modelling",
            "claim_boundary": "not full first-principles Einstein-QCD",
        })
    if aniso is not None:
        ok = aniso[aniso["status"].astype(str).str.startswith("ok")]
        rows.append({
            "module": "Anisotropic Einstein + CFL effective pressures",
            "solutions": len(ok),
            "max_mass_Msun": float(ok["M_msun"].max()) if len(ok) else np.nan,
            "max_compactness": float(ok["compactness_2M_R"].max()) if len(ok) else np.nan,
            "role_in_BEY": "tests anisotropic pressure support and energy-condition compatibility",
            "claim_boundary": "not full relativistic QCD transport or nonlinear stability theorem",
        })

    summary = pd.DataFrame(rows)
    summary.to_csv(results / "einstein_cfl_master_summary.csv", index=False)

    claim_text = """# Einstein-CFL effective-core claim-safety note

Recommended wording:

"We solve the static Einstein equations in TOV form for a compact core sourced by
a QCD-inspired CFL/bag effective equation of state, and extend the calculation
to an anisotropic effective-pressure channel."

Avoid:

"We solve full Einstein-QCD from first principles."

Safe interpretation:

The isotropic and anisotropic TOV modules reduce the gap between a purely
geometric regular-core ansatz and a matter-supported compact-core calculation.
They are effective Einstein+EOS models, not microscopic QCD-in-curved-spacetime
solutions.
"""
    (results / "einstein_cfl_claim_safety_note.md").write_text(claim_text, encoding="utf-8")

    latex = r"""\subsection{Einstein--CFL effective-core solutions}
\label{sec:einstein_cfl_effective_solutions}

As an additional matter-supported check, we solve the static Einstein equations
in TOV form for a compact core sourced by a QCD-inspired CFL/bag effective
equation of state. This calculation should not be read as a first-principles
Einstein--QCD solution. Rather, it tests whether the regular-core picture can be
supported by a relativistic hydrostatic configuration with a dense-matter
effective EOS.

We also consider an anisotropic effective-pressure extension,
\[
p_t(r)=p_r(r)+\alpha p_r(r)\frac{r^2}{r^2+R_a^2},
\]
which vanishes at the centre and allows smooth tangential pressure support in
the outer core. The corresponding anisotropic TOV equation includes the
standard \(2(p_t-p_r)/r\) force term. This branch is used as an effective
pressure-support diagnostic, not as a nonlinear stability theorem or a
microscopic QCD transport calculation.
"""
    (results / "einstein_cfl_latex_snippet.tex").write_text(latex, encoding="utf-8")

    print("Einstein-CFL master summary completed.")
    print(f"Results: {results}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
