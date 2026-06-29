#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run the BEY Schwarzschild Condensation diagnostic pipeline.

This runner preserves the historical script order while giving the GitHub repo
a single entry point. It executes the scripts in numbered order and stops on the
first failure.
"""
from __future__ import annotations

from pathlib import Path
import subprocess
import sys


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    scripts = [
        "01_hesaplamalar.py",
        "02_figure_mass_density.py",
        "03_results_to_csv.py",
        "04_figure_phase_diagram.py",
        "05_figure_desitter_core.py",
        "06_figure_gaussian_profile.py",
        "07_einstein_gp_solver.py",
        "08_stability_analysis.py",
        "09_qnm_spectrum.py",
        "10_timescales.py",
        "11_kerr_extension.py",
        "12_time_domain_echo.py",
        "13_entropy_analysis.py",
        "14_qcd_phase_transition.py",
        "15_energy_conditions.py",
        "16_junction_matching.py",
        "17_causal_tapered_core.py",
        "18_smooth_transition_matching.py",
        "19_curvature_regular_core.py",
        "20_compactness_branch_map.py",
        "21_bey_master_summary.py",
        "22_einstein_qcd_tov_solver.py",
        "23_anisotropic_cfl_core_solver.py",
        "24_einstein_cfl_master_summary.py",
        "25_relativistic_gl_order_parameter.py",
        "26_gl_free_energy_diagnostic.py",
    ]
    for name in scripts:
        script = root / "code" / name
        print("=" * 80)
        print(f"Running {name}")
        print("=" * 80)
        subprocess.check_call([sys.executable, str(script)], cwd=str(root))
    print("BEY Schwarzschild Condensation pipeline completed.")


if __name__ == "__main__":
    main()
