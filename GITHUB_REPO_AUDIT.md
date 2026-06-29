# BEY GitHub Repository Audit

## Executive summary

Creating a GitHub repository for the BEY code is scientifically and practically sensible. The codebase is no longer a loose scratch folder: it contains a numbered reproducibility pipeline, result tables, and publication-style figures.

The cleanup pack removes archive clutter and bytecode, consolidates dependencies, adds a unified runner, and adds repository metadata.

## What was cleaned

- `code/utils/__pycache__/constants.cpython-311.pyc`
- `docker/requirements.txt.txt`
- `figures/fig09_kerr_extension_fixed.pdf`
- `figures/fig09_kerr_extension_fixed.png`
- `results.zip`
- `schwarzschild-condensation.zip`
- `schwarzschild-condensationfour.zip`
- `schwarzschild-condensationone.zip`
- `schwarzschild-condensationthree.zip`
- `schwarzschild-condensationtwo.zip`

## Duplicate status after cleanup

- ['requirements.txt', 'docker/requirements.txt']

## Python static check

- None. All Python files compile syntactically.

## Final repository statistics

- Files: 220
- Total size: 9.32 MB
- Python files checked: 36
- Suffix counts: `{'': 2, '.yml': 1, '.txt': 2, '.md': 16, '.py': 36, '.pdf': 62, '.png': 62, '.csv': 34, '.tex': 5}`

## Recommended GitHub repository name

`bey-schwarzschild-condensation`

Alternative names:

- `BEY-regular-core-diagnostics`
- `schwarzschild-condensation-bey`
- `bey-compact-object-framework`

## Recommendation

Make the repository public only after adding a license decision and a short claim-safety note in the repository description. Suggested safe description:

> Reproducibility archive for BEY Schwarzschild Condensation compact-object diagnostics: regular-core scans, energy conditions, matching, TOV/EOS, QNM/echo-inspired observables, and relativistic GL diagnostics.

Avoid wording like “black hole interior solved” or “definitive model of black holes.”
