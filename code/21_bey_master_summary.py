#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
21_bey_master_summary.py

BEY / Schwarzschild Condensation patch-v4:
Master evidence, claim-safety, figure-selection, and manuscript integration summary.

Purpose
-------
Patch-v1 added energy-condition diagnostics.
Patch-v2 added tapered boundary and smooth-transition diagnostics.
Patch-v3 added curvature regularity and compactness branch maps.

Patch-v4 does not add a new physical diagnostic. It integrates v1-v3 into
paper-ready tables and text snippets.

Outputs
-------
results/BEY_master_evidence_table.csv
results/BEY_claim_safety_table.csv
results/BEY_figures_for_paper.md
results/BEY_results_section_draft.md
results/BEY_latex_results_snippet.tex
results/BEY_final_readiness_checklist.md
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def read_csv_if_exists(path: Path) -> pd.DataFrame | None:
    if path.exists():
        return pd.read_csv(path)
    return None


def value(df: pd.DataFrame | None, col: str, default="not available"):
    if df is None or df.empty or col not in df.columns:
        return default
    x = df.iloc[0][col]
    try:
        if float(x).is_integer():
            return int(x)
    except Exception:
        pass
    return x


def fmt_count(num, den):
    if isinstance(num, str) or isinstance(den, str):
        return "not available"
    return f"{int(num)}/{int(den)}"


def build_master_evidence(results: Path) -> pd.DataFrame:
    e1 = read_csv_if_exists(results / "energy_conditions_summary.csv")
    e2 = read_csv_if_exists(results / "causal_tapered_core_summary.csv")
    e3 = read_csv_if_exists(results / "curvature_regular_core_summary.csv")
    match = read_csv_if_exists(results / "smooth_transition_matching_comparison.csv")

    rows = []

    n1 = value(e1, "n_parameter_sets")
    rows.append({
        "Layer": "Patch v1: untapered energy-condition baseline",
        "Test": "NEC/WEC/SEC/DEC, centre regularity, strict causal proxy",
        "Primary outcome": (
            f"NEC={fmt_count(value(e1,'NEC_pass_count'), n1)}, "
            f"WEC={fmt_count(value(e1,'WEC_pass_count'), n1)}, "
            f"SEC={fmt_count(value(e1,'SEC_pass_count'), n1)}, "
            f"DEC={fmt_count(value(e1,'DEC_pass_count'), n1)}"
        ),
        "Secondary outcome": (
            f"centre regular={fmt_count(value(e1,'center_regular_count'), n1)}, "
            f"strict={fmt_count(value(e1,'strict_accept_count'), n1)}, "
            f"regularizing={fmt_count(value(e1,'regularizing_accept_count'), n1)}"
        ),
        "Paper role": "Baseline regular-core energy-condition diagnostic",
        "Claim strength": "supportive toy-model evidence",
        "Caveat": "Untapered branch may leave nonzero boundary pressure."
    })

    n2 = value(e2, "n_parameter_sets")
    rows.append({
        "Layer": "Patch v2: tapered boundary branch",
        "Test": "Tapered density/pressure, NEC/WEC/SEC/DEC, causality, boundary softness",
        "Primary outcome": (
            f"NEC={fmt_count(value(e2,'NEC_pass_count'), n2)}, "
            f"WEC={fmt_count(value(e2,'WEC_pass_count'), n2)}, "
            f"SEC={fmt_count(value(e2,'SEC_pass_count'), n2)}, "
            f"DEC={fmt_count(value(e2,'DEC_pass_count'), n2)}"
        ),
        "Secondary outcome": (
            f"centre regular={fmt_count(value(e2,'center_regular_count'), n2)}, "
            f"boundary soft={fmt_count(value(e2,'boundary_soft_count'), n2)}, "
            f"strict={fmt_count(value(e2,'strict_accept_count'), n2)}"
        ),
        "Paper role": "Main effective-core branch",
        "Claim strength": "strong toy-model consistency evidence",
        "Caveat": "Causal proxy is still an ansatz-level filter, not full perturbative stability."
    })

    # Matching comparison summaries
    if match is not None and not match.empty and "profile" in match.columns:
        def prof_mean(profile: str, col: str):
            m = match[match["profile"] == profile]
            if m.empty or col not in m.columns:
                return np.nan
            return float(np.mean(np.abs(m[col])))
        def prof_sum(profile: str, col: str):
            m = match[match["profile"] == profile]
            if m.empty or col not in m.columns:
                return np.nan
            return int(m[col].sum())
        unt_p = prof_mean("untapered", "pr_boundary")
        tap_p = prof_mean("tapered", "pr_boundary")
        unt_warn = prof_sum("untapered", "thin_shell_warning")
        tap_warn = prof_sum("tapered", "thin_shell_warning")
        matching_outcome = f"mean |p_r boundary|: untapered={unt_p:.3e}, tapered={tap_p:.3e}; warnings={unt_warn}/12 to {tap_warn}/12"
    else:
        matching_outcome = "not available"

    rows.append({
        "Layer": "Patch v2: smooth-transition matching comparison",
        "Test": "Untapered vs tapered boundary density/pressure and matching warning",
        "Primary outcome": matching_outcome,
        "Secondary outcome": "Metric continuity is enforced by M=m(R_match) in the diagnostic setup.",
        "Paper role": "Boundary-pressure reduction evidence",
        "Claim strength": "strong diagnostic improvement",
        "Caveat": "Not a complete Darmois-Israel junction calculation."
    })

    n3 = value(e3, "n_parameter_sets")
    rows.append({
        "Layer": "Patch v3: curvature regularity",
        "Test": "Ricci scalar proxy, Kretschmann-like proxy, regular-center proxy",
        "Primary outcome": (
            f"finite curvature={fmt_count(value(e3,'finite_curvature_count'), n3)}, "
            f"regular centre={fmt_count(value(e3,'regular_center_proxy_count'), n3)}"
        ),
        "Secondary outcome": (
            f"tapered={value(e3,'tapered_count')}, "
            f"untapered={value(e3,'untapered_count')}"
        ),
        "Paper role": "Finite-curvature support for singularity-replacement ansatz",
        "Claim strength": "supportive toy-model evidence",
        "Caveat": "Curvature proxies are computed for an effective single-function metric."
    })

    rows.append({
        "Layer": "Patch v3: compactness branch map",
        "Test": "C=2M/R_match branch classification",
        "Primary outcome": (
            f"subcompact={value(e3,'subcompact_count')}, "
            f"compact below horizon={value(e3,'compact_below_horizon_count')}, "
            f"horizon-near ECO={value(e3,'horizon_near_ECO_count')}, "
            f"static caution={value(e3,'horizon_interior_static_caution_count')}"
        ),
        "Secondary outcome": "Separates the main regular-core branch from the conditional ECO/echo branch.",
        "Paper role": "Prevents claim overreach and branch confusion",
        "Claim strength": "classification diagnostic",
        "Caveat": "C>=1 branch requires dynamic/interior treatment beyond the static ansatz."
    })

    return pd.DataFrame(rows)


def build_claim_safety_table() -> pd.DataFrame:
    rows = [
        {
            "Claim": "The BEY ansatz can replace the classical central divergence by a finite-density effective core at toy-model level.",
            "Supported by": "regular centre, m(r) behavior, finite curvature diagnostics",
            "Allowed wording": "supports a finite-curvature regular-core replacement at the toy-model level",
            "Forbidden wording": "proves the black-hole singularity is solved",
            "Status": "safe with caveat",
        },
        {
            "Claim": "The tapered branch improves matching to a Schwarzschild exterior.",
            "Supported by": "boundary density/pressure suppression and reduced matching warnings",
            "Allowed wording": "reduces boundary-pressure tension and gives a cleaner matching diagnostic",
            "Forbidden wording": "proves a full smooth Darmois-Israel matching",
            "Status": "safe with caveat",
        },
        {
            "Claim": "Energy conditions are compatible with a broad parameter region.",
            "Supported by": "NEC/WEC/SEC/DEC scans",
            "Allowed wording": "the scanned tapered branch satisfies NEC/WEC/SEC/DEC over the tested grid",
            "Forbidden wording": "all physically possible cores satisfy all energy conditions",
            "Status": "safe with scan range specified",
        },
        {
            "Claim": "The model can have an ECO/echo phenomenological branch.",
            "Supported by": "compactness branch map and previous echo modules",
            "Allowed wording": "a horizon-near branch may be used for conditional ECO/echo phenomenology",
            "Forbidden wording": "BEY necessarily predicts echoes",
            "Status": "conditional only",
        },
        {
            "Claim": "The model is QCD-inspired.",
            "Supported by": "effective condensate profile and QCD phase-transition diagnostics",
            "Allowed wording": "QCD-inspired effective condensate-core ansatz",
            "Forbidden wording": "full Einstein-QCD solution",
            "Status": "safe as inspiration, not derivation",
        },
        {
            "Claim": "The model is stable.",
            "Supported by": "toy stability and causal-proxy filters",
            "Allowed wording": "passes first stability diagnostics / causal-proxy filters",
            "Forbidden wording": "nonlinearly stable compact object solution",
            "Status": "not fully proven",
        },
    ]
    return pd.DataFrame(rows)


def write_figures_for_paper(path: Path) -> None:
    text = """# BEY figures recommended for the paper

## Main text figures

1. `fig15a_tapered_density.pdf`  
   Use in the effective-core diagnostics subsection. Shows the tapered profile and the matching surface.

2. `fig15d_tapered_nec_dec.pdf`  
   Use with the energy-condition discussion. Shows NEC/DEC positivity in the fiducial branch.

3. `fig16a_density_boundary_comparison.pdf` and `fig16b_pressure_boundary_comparison.pdf`  
   Use as a two-panel matching diagnostic. Shows why the tapered branch improves boundary behavior.

4. `fig17b_ricci_proxy.pdf` and `fig17c_kretschmann_proxy.pdf`  
   Use in the curvature regularity subsection. Shows finite curvature diagnostics.

5. `fig18a_compactness_branch_map.pdf`  
   Use in the compactness branch subsection. Separates subcompact, horizon-near, and static-caution branches.

## Appendix figures

- `fig15f_tapered_causality.pdf`
- `fig15h_tapered_strict_acceptance.pdf`
- `fig15i_tapered_regularizing_acceptance.pdf`
- `fig17d_curvature_log.pdf`
- `fig17f_curvature_vs_compactness.pdf`
- `fig18b_branch_counts.pdf`

## Figure-use caution

Do not present the echo/compactness branch as mandatory. It is a conditional phenomenological branch.
"""
    path.write_text(text, encoding="utf-8")


def write_results_section(path: Path, evidence: pd.DataFrame) -> None:
    text = r"""# BEY results-section draft

## Effective-core diagnostics

We tested the effective regular-core ansatz through a sequence of diagnostic scans. The untapered baseline scan contains 60 parameter sets. All tested sets have a regular centre; 56 satisfy NEC, WEC, and DEC; 32 satisfy SEC; 8 pass the strict causal-proxy filter; and 56 pass the broader regularizing-core filter. This baseline branch is useful for diagnosing the regular-core mechanism, but it generically leaves a nonzero effective boundary pressure at finite matching radius.

To reduce this matching tension, we introduced a tapered branch in which the density and effective pressures vanish smoothly at the matching surface. In the 288-set tapered scan, all parameter sets satisfy NEC, WEC, SEC, DEC, centre regularity, and boundary softness; 240 pass the strict causal-proxy filter. In the matching comparison, the mean absolute boundary pressure is reduced from \(1.77\times10^{-3}\) in the untapered branch to \(1.92\times10^{-10}\) in the tapered branch, and the thin-shell warning count is reduced from \(12/12\) to \(0/12\). This supports the tapered branch as the preferred effective-core ansatz for the main text.

## Curvature and compactness diagnostics

We further tested the regularity of the tapered-core branch by evaluating curvature diagnostics associated with the effective single-function metric. The Ricci-scalar and Kretschmann-like proxies remain finite over the scanned regularized domain, with 96/96 cases passing finite-curvature and regular-centre proxy checks. These results support the interpretation of the ansatz as a finite-curvature replacement of the classical central singularity at the toy-model level.

A compactness scan based on \(C=2M/R_{\rm match}\) separates subcompact, compact-below-horizon, horizon-near ECO, and horizon-interior/static-coordinate branches. This classification is used only as a diagnostic separation. The \(C\geq1\) branch is treated with caution because the present static single-function ansatz does not constitute a complete dynamical black-hole interior construction.

## Safe conclusion paragraph

Taken together, the energy-condition, tapered-boundary, curvature, and compactness diagnostics support BEY as a regular-core singularity-replacement framework at the effective toy-model level. The preferred branch is a tapered, positive-pressure regular core that preserves centre regularity, satisfies the scanned energy-condition filters, suppresses residual boundary pressure at the matching surface, and yields finite curvature proxies. The framework should not be interpreted as a full Einstein-QCD solution, a complete Darmois-Israel matching construction, or a nonlinear stability proof; rather, it provides a controlled phenomenological ansatz for replacing the classical central singularity with a finite-density effective condensate core.
"""
    path.write_text(text, encoding="utf-8")


def write_latex_snippet(path: Path) -> None:
    text = r"""\subsection{Master diagnostic summary}
\label{subsec:master_diagnostic_summary}

The diagnostic pipeline developed above can be summarized as a three-stage
consistency test. First, the untapered regular-core baseline establishes centre
regularity and broad compatibility with the standard energy-condition filters.
Second, the tapered branch suppresses residual boundary density and pressure at
the matching surface, thereby reducing the matching tension that appears in the
untapered branch. Third, curvature and compactness diagnostics test whether the
effective single-function metric remains finite in the regularized core and
separate subcompact, horizon-near, and static-coordinate caution branches.

The preferred fiducial branch is the tapered positive-pressure core
\[
(A,w,\alpha,x_m,q)=(1.0,0.3,0.0,4.0,2),
\]
which is used only as a representative toy-model point. It should not be
interpreted as a unique astrophysical solution. Within the scanned grid, the
tapered branch satisfies NEC, WEC, SEC, DEC, centre regularity, and boundary
softness in all tested cases, while the strict causal-proxy filter selects
240 out of 288 cases. The curvature scan gives finite Ricci-scalar and
Kretschmann-like proxies in all 96 tested cases.

These results support the use of BEY as an effective singularity-replacement
compact-core ansatz. They do not constitute a full Einstein--QCD solution, a
complete Darmois--Israel matching proof, or a nonlinear stability theorem.
"""
    path.write_text(text, encoding="utf-8")


def write_readiness_checklist(path: Path) -> None:
    text = """# BEY final readiness checklist

## Completed diagnostic layers

- [x] Untapered energy-condition baseline
- [x] Tapered boundary branch
- [x] Smooth-transition matching comparison
- [x] Curvature regularity diagnostics
- [x] Compactness branch map
- [x] Claim-safety table
- [x] Paper figure selection list
- [x] Results-section draft

## Still not claimed

- [ ] Full Einstein-QCD derivation
- [ ] Full Darmois-Israel junction proof
- [ ] Full nonlinear stability theorem
- [ ] Mandatory echo prediction
- [ ] Unique physical black-hole interior solution

## Recommended next manuscript action

Integrate:
- `BEY_master_evidence_table.csv`
- `BEY_claim_safety_table.csv`
- `BEY_results_section_draft.md`
- `BEY_latex_results_snippet.tex`

into the BEY manuscript as a short diagnostic summary subsection and one appendix table.
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    root = project_root()
    results = root / "results"
    results.mkdir(exist_ok=True)

    evidence = build_master_evidence(results)
    claim_safety = build_claim_safety_table()

    evidence.to_csv(results / "BEY_master_evidence_table.csv", index=False)
    claim_safety.to_csv(results / "BEY_claim_safety_table.csv", index=False)

    write_figures_for_paper(results / "BEY_figures_for_paper.md")
    write_results_section(results / "BEY_results_section_draft.md", evidence)
    write_latex_snippet(results / "BEY_latex_results_snippet.tex")
    write_readiness_checklist(results / "BEY_final_readiness_checklist.md")

    print("BEY master integration summary completed.")
    print(f"Results: {results}")
    print("Created:")
    for name in [
        "BEY_master_evidence_table.csv",
        "BEY_claim_safety_table.csv",
        "BEY_figures_for_paper.md",
        "BEY_results_section_draft.md",
        "BEY_latex_results_snippet.tex",
        "BEY_final_readiness_checklist.md",
    ]:
        print(f" - {results / name}")


if __name__ == "__main__":
    main()
