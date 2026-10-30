# How this code was checked against the paper code

The paper model was trained and exported with a larger research code base. This repository is a rewrite that keeps
only what the paper model uses. It was compared with the original code in four ways; the comparison scripts and raw
results are kept with the original code, not here.

| check | what was compared | criterion | result |
|---|---|---|---|
| model | new `CatalogEncoder` + heads with converted paper weights vs the original model, same synthetic inputs (hydro, N-body, masked groups, padding, edge cases), CPU | identical outputs | identical: max difference 0.0 in float32 and float64 |
| data | `read_catalog` vs the original pipeline on 18 catalogs (LH, CV, 1P of TNG/SIMBA/Astrid, Swift-EAGLE, N-body twins) | same subhalos and flags; values equal up to float32 rounding | flags identical; values differ by at most 9.5e-7 (float32 rounding of `log10`, which depends on the CPU's vector instructions) |
| embedding | `embed` with the released weights vs the paper's exported embeddings, all 2,400 LH train and validation catalogs | relative L2 difference <= 1e-4 per catalog | max 5.6e-5, median 1.4e-6; subhalo counts identical |
| training | new training code started from the paper run's initial state vs the paper run, first 500 updates (50-update moving averages) | gap <= 5 % | max gap: M 1.3 %, O 3.4 %, total 1.2 % |

A new run from a fresh initialisation (`scripts/init_model.py`, seed 1) was also trained for 2,000 updates. Its
total loss stays within 3.5 % of the paper run (M within 2.7 %); its teacher-prediction loss O is up to 12 % *lower*
than the paper run's between updates 200 and 2,000. The paper run's initial weights were drawn by an older version
of the code and cannot be regenerated, so a new run reproduces the paper model statistically, not bit for bit.

Two kinds of differences from the original code are deliberate and do not change the method:

* training views and masks are drawn from different random streams (one generator per update instead of one per
  catalog), so individual updates differ while the distribution of views and masks is the same;
* the input `log10_subhalo_total_mass_1e10_msun_per_h`, which the paper model never reads (it was always masked), is
  removed; the converted weights are adjusted so that the encoder's outputs are unchanged.
