#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_all_bey_patch_v6.py

Runs BEY patch-v6:
- 22_einstein_qcd_tov_solver.py
- 23_anisotropic_cfl_core_solver.py
- 24_einstein_cfl_master_summary.py
"""

from pathlib import Path
import subprocess
import sys


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    scripts = [
        root / "code" / "22_einstein_qcd_tov_solver.py",
        root / "code" / "23_anisotropic_cfl_core_solver.py",
        root / "code" / "24_einstein_cfl_master_summary.py",
    ]
    for script in scripts:
        print("=" * 72)
        print(f"Running {script.name}")
        print("=" * 72)
        subprocess.check_call([sys.executable, str(script)])
    print("BEY patch-v6 Einstein-CFL diagnostics completed.")


if __name__ == "__main__":
    main()
