#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_all_bey_patch_v2.py

Runs BEY patch-v2:
- 17_causal_tapered_core.py
- 18_smooth_transition_matching.py
"""

from pathlib import Path
import subprocess
import sys


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    scripts = [
        root / "code" / "17_causal_tapered_core.py",
        root / "code" / "18_smooth_transition_matching.py",
    ]
    for script in scripts:
        print("=" * 72)
        print(f"Running {script.name}")
        print("=" * 72)
        subprocess.check_call([sys.executable, str(script)])
    print("BEY patch-v2 diagnostics completed.")


if __name__ == "__main__":
    main()
