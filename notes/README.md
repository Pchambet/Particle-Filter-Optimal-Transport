# Optimal transport notes (NAIST, May 2025)

Written for a workshop of the Computational Systems Biology Laboratory at NAIST
(Nara Institute of Science and Technology), May 2025.

| File | Content |
| --- | --- |
| [`ot-foundations.pdf`](ot-foundations.pdf) ([source](ot-foundations.tex)) | *Foundations of Optimal Transport*, 18 pages: measures and pushforwards, Monge and Kantorovich, duality, Wasserstein distances, Brenier's theorem, and the saddle-point method for learning OT maps. |
| [`ot-maps-paper-review.pdf`](ot-maps-paper-review.pdf) ([source](ot-maps-paper-review.tex)) | Review of Rout, Korotin & Burnaev, *Generative Modeling with Optimal Transport Maps* (ICLR 2022): AE-OT vs OT-map, and the derivation of the min-max objective from the Kantorovich dual. |

Revised in 2026: typos, and a corrected statement of the quadratic-cost dual.

Build: `latexmk -pdf <name>.tex` (needs `tcolorbox` with its `skins` library, `physics`, `mathtools`, `microtype`).
