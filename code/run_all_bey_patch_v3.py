#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_all_bey_patch_v3.py

Runs BEY patch-v3:
- 19_curvature_regular_core.py
- 20_compactness_branch_map.py
"""

from pathlib import Path
import subprocess
import sys


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    scripts = [
        root / "code" / "19_curvature_regular_core.py",
        root / "code" / "20_compactness_branch_map.py",
    ]
    for script in scripts:
        print("=" * 72)
        print(f"Running {script.name}")
        print("=" * 72)
        subprocess.check_call([sys.executable, str(script)])
    print("BEY patch-v3 diagnostics completed.")


if __name__ == "__main__":
    main()
