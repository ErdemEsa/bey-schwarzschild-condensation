#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_all_bey_patch_v4.py

Runs BEY patch-v4:
- 21_bey_master_summary.py
"""

from pathlib import Path
import subprocess
import sys


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    script = root / "code" / "21_bey_master_summary.py"
    print("=" * 72)
    print(f"Running {script.name}")
    print("=" * 72)
    subprocess.check_call([sys.executable, str(script)])
    print("BEY patch-v4 integration completed.")


if __name__ == "__main__":
    main()
