# DiT capacity/data comparison — Session B

This report uses frozen R4 state/detail codec/statistics, factorized_v2, 16-frame dt=100 ps clips, and real per-clip generation. Test coordinates were not opened.

## Main final comparison

| ID | source | depth | data | H | region | aligned RMSD | dRMSD | bond RMSE | contact F1 | RMSF ratio |
|---|---|---:|---|---:|---|---:|---:|---:|---:|---:|
| G48 | gaussian | 4 | base48 | 4 | L4 | 3.29934 | 2.73945 | 2.68524 | 0.439623 | 1.26692 |
| G48 | gaussian | 4 | base48 | 4 | L8 | 3.375 | 2.76053 | 2.71598 | 0.43554 | 2.11022 |
| G48 | gaussian | 4 | base48 | 4 | future | 3.44597 | 2.78251 | 2.74403 | 0.431914 | 2.23381 |
| G48 | gaussian | 4 | base48 | 8 | L4 | 3.23802 | 2.79048 | 2.61145 | 0.453102 | 1.21658 |
| G48 | gaussian | 4 | base48 | 8 | L8 | 3.3168 | 2.81133 | 2.62568 | 0.450369 | 2.09353 |
| G48 | gaussian | 4 | base48 | 8 | future | 3.3168 | 2.81133 | 2.62568 | 0.450369 | 2.09353 |
| C48 | conditional | 4 | base48 | 4 | L4 | 3.75251 | 2.99507 | 3.51845 | 0.358362 | 1.22366 |
| C48 | conditional | 4 | base48 | 4 | L8 | 3.83519 | 3.04635 | 3.52335 | 0.357034 | 2.68568 |
| C48 | conditional | 4 | base48 | 4 | future | 3.90952 | 3.08872 | 3.52693 | 0.3562 | 2.90802 |
| C48 | conditional | 4 | base48 | 8 | L4 | 3.76965 | 3.00541 | 3.53156 | 0.35798 | 1.28451 |
| C48 | conditional | 4 | base48 | 8 | L8 | 3.86386 | 3.06654 | 3.53398 | 0.356633 | 2.72929 |
| C48 | conditional | 4 | base48 | 8 | future | 3.86386 | 3.06654 | 3.53398 | 0.356633 | 2.72929 |
| C192 | conditional | 4 | expanded192 | 4 | L4 | BLOCKED_DATA | n/a | n/a | n/a | n/a |
| C192 | conditional | 4 | expanded192 | 4 | L8 | BLOCKED_DATA | n/a | n/a | n/a | n/a |
| C192 | conditional | 4 | expanded192 | 4 | future | BLOCKED_DATA | n/a | n/a | n/a | n/a |
| C192 | conditional | 4 | expanded192 | 8 | L4 | BLOCKED_DATA | n/a | n/a | n/a | n/a |
| C192 | conditional | 4 | expanded192 | 8 | L8 | BLOCKED_DATA | n/a | n/a | n/a | n/a |
| C192 | conditional | 4 | expanded192 | 8 | future | BLOCKED_DATA | n/a | n/a | n/a | n/a |
| C48D8 | conditional | 8 | base48 | 4 | L4 | 3.80675 | 3.03953 | 3.51583 | 0.354457 | 1.50708 |
| C48D8 | conditional | 8 | base48 | 4 | L8 | 3.8642 | 3.07122 | 3.5165 | 0.35484 | 2.77735 |
| C48D8 | conditional | 8 | base48 | 4 | future | 3.91644 | 3.09527 | 3.5155 | 0.35556 | 2.95486 |
| C48D8 | conditional | 8 | base48 | 8 | L4 | 3.77899 | 3.01019 | 3.52275 | 0.356918 | 1.40807 |
| C48D8 | conditional | 8 | base48 | 8 | L8 | 3.85018 | 3.0531 | 3.51827 | 0.357276 | 2.75345 |
| C48D8 | conditional | 8 | base48 | 8 | future | 3.85018 | 3.0531 | 3.51827 | 0.357276 | 2.75345 |

## Future motion decomposition

| ID | H | RMSF prediction | RMSF MD | ratio | atom correlation | within-block pred/MD/ratio | between-block pred/MD/ratio |
|---|---:|---:|---:|---:|---:|---|---|
| G48 | 4 | 2.09196 | 0.995433 | 2.23381 | 0.196495 | 2.51668/0.641016/3.9902 | 1.71479/1.63989/1.12639 |
| G48 | 8 | 1.81481 | 0.918732 | 2.09353 | 0.15926 | 2.41672/0.614217/3.99047 | 1.7751/1.55012/1.22952 |
| C48 | 4 | 2.72348 | 0.995433 | 2.90802 | 0.123751 | 2.97043/0.641016/4.7144 | 1.51514/1.63989/1.01829 |
| C48 | 8 | 2.36619 | 0.918732 | 2.72929 | 0.133346 | 2.97122/0.614217/4.91045 | 1.53492/1.55012/1.08517 |
| C192 | 4 | BLOCKED_DATA | n/a | n/a | n/a | n/a | n/a |
| C192 | 8 | BLOCKED_DATA | n/a | n/a | n/a | n/a | n/a |
| C48D8 | 4 | 2.76702 | 0.995433 | 2.95486 | 0.126379 | 2.9953/0.641016/4.75378 | 1.60543/1.63989/1.0788 |
| C48D8 | 8 | 2.38675 | 0.918732 | 2.75345 | 0.121263 | 2.99312/0.614217/4.94665 | 1.58052/1.55012/1.1169 |

## Comparison decisions

The source-gain criterion requires lower aligned RMSD, lower bond RMSE, and higher contact F1 after averaging the system-equal H4/H8 future metrics; no RF-loss cross-source ranking or hidden composite is used.

- Conditional source gain retained (C48 vs G48): NO — none of the three primary geometry/contact metrics improves; delta(candidate-baseline) aligned RMSD=0.505305, bond RMSE=0.845602, contact F1=-0.0847254.
- More training systems (C192 vs C48): BLOCKED_DATA — expanded192 source is incomplete on this machine: 26784 selected clips are missing
- More DiT layers (C48D8 vs C48): YES — all three primary geometry/contact metrics improve; delta(candidate-baseline) aligned RMSD=-0.00338106, bond RMSE=-0.013577, contact F1=1.85648e-06.

- Depth magnitude/motion caveat: the directional depth deltas are negligible (contact F1 +1.86e-6), while future RMSF ratio rises from 2.81866 to 2.85416 and within-block displacement ratio rises from 4.81243 to 4.85022; this is not evidence of a substantial motion-quality gain.
Data-versus-depth cannot be ranked when either C192 or C48D8 is incomplete; completed deltas above remain valid without filling the missing arm.

## Cost and training-system geometry

| ID | status | updates | tokens | training GPU-hours | evaluation GPU-hours | train-subset future bond RMSE |
|---|---|---:|---:|---:|---:|---:|
| G48 | PASS | 4500 | 267988032 | ~3.22136* | 0.675119 | 2.68525 |
| C48 | PASS | 4500 | 267988032 | 4.94132 | 0.690703 | 3.53405 |
| C192 | BLOCKED_DATA | n/a | n/a | n/a | n/a | n/a |
| C48D8 | PASS | 4500 | 267988032 | 4.99842 | 0.71213 | 3.52099 |

Frozen conditional-source template bond RMSE reference: 0.0127156 A.
*G48 cost audit: its `train_summary.json` covers only the resumed segment (2.35693 GPU-hours). The initial formal 0→2000-step attempt ran approximately 0.86444 wall-hours before the CUSOLVER monitor failure; the separate precheckpoint step-89 smoke was excluded. Thus all formal training attempts are estimated at 13.16110 GPU-hours across the three completed arms. Generation evaluation totals 2.07795 GPU-hours, for approximately 15.23906 GPU-hours of training plus real-generation evaluation, excluding CPU-only preparation and targeted-test overhead. This remains within the frozen 32 GPU-hour budget.
All completed validation arms exceed 10x that frozen-source bond reference. Together with path RMSD/dRMSD and within/between-block motion errors above, this supports testing explicit geometry supervision or local atom interactions next; neither was added in this phase.

## Interpretation rules

- G48 vs C48 isolates source mode at the original 48-system capacity.
- C48 vs C192 isolates nested training-system diversity at depth 4; the data manifest proves system disjointness, but no sequence/homology audit was available, so no family-generalization claim is made.
- C48 vs C48D8 isolates DiT depth at the original 48-system data scale.
- RF validation is reported within source/configuration and is not used to rank Gaussian against Conditional.
- ACF/Pearson-like values are null for constant or insufficient signals; missing torsion indices remain null.
- Legacy per-frame occupancy and unimplemented diversity are excluded from formal aggregates. The separately named true-future occupancy MAE is null when its all-pair implementation is not applicable.

Code commit: `9960a479071ddd50df575790a2fb6388fa399e93`
Output root: `/workspace/molvid-dit-capacity-data-v1/outputs/dit_capacity_data_v1/20260914_capacity_data_v1`
