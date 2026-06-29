#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run BEY patch-v7 relativistic GL diagnostics."""
from pathlib import Path
import subprocess
import sys


def main():
    root = Path(__file__).resolve().parents[1]
    scripts = [
        root / "code" / "25_relativistic_gl_order_parameter.py",
        root / "code" / "26_gl_free_energy_diagnostic.py",
    ]
    for script in scripts:
        print("=" * 72)
        print(f"Running {script.name}")
        print("=" * 72)
        subprocess.check_call([sys.executable, str(script)])
    print("BEY patch-v7 relativistic GL diagnostics completed.")


if __name__ == "__main__":
    main()
