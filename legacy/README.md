# Legacy code (2023-2024 student project)

This folder keeps the original code and notebooks of the Cassiopée project
(Télécom SudParis, 2023-2024) unchanged, for the record. **It is superseded by the
PyTorch package in [`src/otpf/`](../src/otpf/)**, which is tested against a Kalman
filter ground truth.

A later review found that the legacy code does not demonstrate what the original
README claimed:

| File | Issue |
| --- | --- |
| `code/optimal_transport.py` | `optimal_transport_resampling` computes a Sinkhorn plan `G` but then draws indices from `G.sum(axis=1)`, which equals the particle weights. The "OT" filter is therefore multinomial resampling with extra work, and no barycentric (transport) update is ever applied. |
| `code/optimal_transport.py` | The Sinkhorn solver is `ot.sinkhorn` (POT, NumPy): nothing in the pipeline is differentiated. |
| `code/auto_differentiation.py` | `transition()` receives `Q` but uses a noise array pre-drawn with the initial `Q`, so the gradient with respect to `Q` is identically zero. |
| `code/auto_differentiation.py` | The cost function bypasses resampling entirely and the demo uses systematic resampling, so it says nothing about differentiating through OT resampling. |
| `images/cmse_comparison.png` | The "OT filter is more accurate" curve comes from a single unseeded run of 30 steps. Over many seeds the two filters are statistically indistinguishable on this model (see the main README, experiment 4). |

The rebuilt package fixes each point: noise is reparameterised so gradients reach
every parameter, resampling is a differentiable log-domain Sinkhorn barycentric
projection, and every claim is checked over many seeds against the exact Kalman
log-likelihood and its gradient.
