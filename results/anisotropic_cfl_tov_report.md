# Anisotropic Einstein-CFL effective-core solver report

EOS: B=80 MeV/fm^3, Delta=50 MeV
Alpha values scanned: [-0.2, -0.1, 0.0, 0.1, 0.2, 0.3]
Central densities per alpha: 37
Successful anisotropic solutions: 222

Boundary:
This is an anisotropic Einstein + CFL-inspired effective-pressure model, not a
full relativistic QCD transport or nonlinear stability calculation.

Best rows by alpha:
 eps_c_MeVfm3  B_MeVfm3  Delta_MeV  epsilon_surface_MeVfm3  alpha  R_aniso_km   M_msun   R_km  compactness_2M_R  max_sigma_over_pr  NEC_pass  SEC_pass  DEC_pass  n_steps status
       1150.0        80         50              272.527364    0.3         8.0 1.950153 10.275          0.560532       1.867762e-01      True      True      True      411     ok
       1150.0        80         50              272.527364    0.2         8.0 1.913666 10.250          0.551386       1.242884e-01      True      True      True      410     ok
       1200.0        80         50              272.527364    0.1         8.0 1.877527 10.175          0.544961       6.179803e-02      True      True      True      407     ok
       1250.0        80         50              272.527364    0.0         8.0 1.841562 10.100          0.538491       0.000000e+00      True      True      True      404     ok
       1400.0        80         50              272.527364   -0.1         8.0 1.807396  9.950          0.536468      -9.765530e-07      True      True      True      398     ok
       1500.0        80         50              272.527364   -0.2         8.0 1.772709  9.850          0.531514      -1.953106e-06      True      True      True      394     ok
