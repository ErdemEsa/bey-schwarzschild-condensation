#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_all_bey_patch_v1.py

Runs the new BEY patch-v1 diagnostics:
- 15_energy_conditions.py
- 16_junction_matching.py
"""

from pathlib import Path
import subprocess
import sys


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    scripts = [
        root / "code" / "15_energy_conditions.py",
        root / "code" / "16_junction_matching.py",
    ]
    for script in scripts:
        print("=" * 72)
        print(f"Running {script.name}")
        print("=" * 72)
        subprocess.check_call([sys.executable, str(script)])
    print("BEY patch-v1 diagnostics completed.")


if __name__ == "__main__":
    main()
