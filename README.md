# BEY Schwarzschild Condensation

Finite-volume and phenomenological diagnostics for the BEY compact-object / regular-core research programme.

This repository contains the Python scripts, generated CSV tables, and publication-style figures used for the BEY Schwarzschild Condensation analysis. The code is organized as a reproducibility archive: scripts are intentionally numbered in the approximate order in which the diagnostics were built.

## Scientific scope

The repository supports a phenomenological compact-object model involving regular-core diagnostics, energy-condition scans, matching analyses, echo/QNM-inspired observables, QCD/CFL-motivated equations of state, anisotropic TOV branches, and relativistic Ginzburg-Landau diagnostics.

This code does **not** claim to prove a final black-hole interior model. It provides a reproducible numerical and figure-generation layer for a BEY-style regular-core / compact-object research programme.

## Repository layout

```text
code/                         Main numbered analysis scripts
code/utils/                   Shared constants and plotting helpers
results/                      Generated CSV, TeX, and Markdown result tables
figures/                      Figures 1-12
fig13_energy_conditions/      Energy-condition figures
fig14_junction_matching/      Junction/matching figures
fig15_causal_tapered_core/    Causal tapered core figures
fig16_smooth_transition_matching/
fig17_curvature_regular_core/
fig18_compactness_branch_map/
fig19_einstein_qcd_tov/
fig20_anisotropic_cfl_core/
fig21_relativistic_gl/
docker/                       Docker requirements
tests/                        Static repository checks
```

## Quick start

### Local Python

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python code/run_all_bey.py
```

### Docker

```bash
docker compose build
docker compose run --rm bey-compute python code/run_all_bey.py
```

### Static checks

```bash
pytest
```

## Main script map

| Script | Purpose |
|---|---|
| `01_hesaplamalar.py` | Baseline Schwarzschild and density calculations |
| `02_figure_mass_density.py` | Mass-density figure generation |
| `03_results_to_csv.py` | CSV and LaTeX table generation |
| `04_figure_phase_diagram.py` | Phase diagram diagnostics |
| `05_figure_desitter_core.py` | de Sitter core metric figure |
| `06_figure_gaussian_profile.py` | Gaussian condensate profile |
| `07_einstein_gp_solver.py` | Einstein-Gross-Pitaevskii toy solver |
| `08_stability_analysis.py` | Linear stability diagnostics |
| `09_qnm_spectrum.py` | QNM-inspired spectrum diagnostics |
| `10_timescales.py` | Condensation/collapse timescales |
| `11_kerr_extension.py` | Slow-rotation/Kerr extension diagnostics |
| `12_time_domain_echo.py` | Echo waveform toy simulation |
| `13_entropy_analysis.py` | Entropy diagnostics |
| `14_qcd_phase_transition.py` | QCD phase-transition diagnostic |
| `15_energy_conditions.py` | NEC/SEC/DEC/causality scans |
| `16_junction_matching.py` | Interior/exterior matching diagnostics |
| `17_causal_tapered_core.py` | Causal tapered-core scan |
| `18_smooth_transition_matching.py` | Smooth boundary/matching comparison |
| `19_curvature_regular_core.py` | Regular-core curvature proxies |
| `20_compactness_branch_map.py` | Compactness branch map |
| `21_bey_master_summary.py` | Master evidence summary |
| `22_einstein_qcd_tov_solver.py` | QCD-inspired TOV scan |
| `23_anisotropic_cfl_core_solver.py` | Anisotropic CFL core scan |
| `24_einstein_cfl_master_summary.py` | Einstein-CFL summary layer |
| `25_relativistic_gl_order_parameter.py` | Relativistic GL order-parameter diagnostics |
| `26_gl_free_energy_diagnostic.py` | GL free-energy diagnostic |

## Citation

Use `CITATION.cff` as a starting point. Update the version, DOI, and paper title when the associated manuscript/preprint is frozen.

## License

No open-source license has been selected in this cleanup pack. Add a license before making the repository public if you want others to reuse the code under explicit terms.
