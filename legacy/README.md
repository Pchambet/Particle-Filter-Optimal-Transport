# Legacy code (2023-2024 student project)

This folder keeps the original code and notebooks of the Cassiopée project
(Télécom SudParis, 2023-2024) unchanged, for the record. **It is superseded by the
PyTorch package in [`src/otpf/`](../src/otpf/)**, which is tested against a Kalman
filter ground truth.

This code and the reports in [`docs/cassiopee-2024/`](../docs/cassiopee-2024/) are the
joint work of the 2023-2024 team (Mael Spangenberger, Léo Ritchie, Pierre Chambet;
supervised by Yohan Petetin). They are included for reference and are not covered by the
repository's MIT licence, which applies to the rebuilt package and the notes.

A later review found that the legacy code does not demonstrate what the original
README claimed:

| File | Issue |
| --- | --- |
| `code/optimal_transport.py` | `optimal_transport_resampling` computes a Sinkhorn plan `G` but then draws indices from `G.sum(axis=1)`, which equals the particle weights. The "OT" filter is therefore multinomial resampling with extra work, and no barycentric (transport) update is ever applied. |
| `code/optimal_transport.py` | The Sinkhorn solver is `ot.sinkhorn` (POT, NumPy): nothing in the pipeline is differentiated. |
| `code/auto_differentiation.py` | `transition()` receives `Q` but uses a noise array pre-drawn with the initial `Q`, so the gradient with respect to `Q` is identically zero. |
| `code/auto_differentiation.py` | The cost function bypasses resampling entirely and the demo uses systematic resampling, so it says nothing about differentiating through OT resampling. |
| `images/cmse_comparison.png` | The "OT filter is more accurate" curve comes from a single unseeded run of 30 steps. Re-tested on a corrected setup (standard noise variances 10 and 1, where the legacy code passed 10 as a standard deviation; a true barycentric OT step; N = 100 particles; 100 seeded sequences of 50 steps), OT resampling is not significantly more accurate at any eps, and is borderline worse at eps = 1 (+0.35 ± 0.34, 95% CI just excluding 0, uncorrected for multiple comparisons; main README, experiment 4). |

The rebuilt package fixes each point: noise is reparameterised so gradients reach
every parameter, resampling is a differentiable log-domain Sinkhorn barycentric
projection, and every claim is checked over many seeds against the exact Kalman
log-likelihood and its gradient.
