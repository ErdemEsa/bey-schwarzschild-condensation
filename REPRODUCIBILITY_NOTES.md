# Reproducibility notes

This repository is arranged as a reproducibility archive rather than a polished Python package.

Recommended workflow:

1. Create a fresh environment from `requirements.txt`.
2. Run `pytest` for static syntax checks.
3. Run individual numbered scripts when debugging.
4. Run `python code/run_all_bey.py` for the full pipeline.
5. Compare regenerated CSV/figure outputs with the included `results/` and figure folders.

Generated figures and result tables are included because they document the state of the BEY evidence package at cleanup time.
