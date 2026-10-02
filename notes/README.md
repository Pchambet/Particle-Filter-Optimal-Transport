# Optimal transport notes (NAIST, May 2025)

Written for a workshop of the Computational Systems Biology Laboratory at NAIST
(Nara Institute of Science and Technology), May 2025.

| File | Content |
| --- | --- |
| [`ot-foundations.pdf`](ot-foundations.pdf) ([source](ot-foundations.tex)) | *Foundations of Optimal Transport*, 18 pages: measures and pushforwards, Monge and Kantorovich, duality, Wasserstein distances, Brenier's theorem, and the saddle-point method for learning OT maps. |
| [`ot-maps-paper-review.pdf`](ot-maps-paper-review.pdf) ([source](ot-maps-paper-review.tex)) | Review of Rout, Burnaev & Korotin, *Generative Modeling with Optimal Transport Maps* (ICLR 2022): AE-OT vs OT-map, and the derivation of the min-max objective from the Kantorovich dual. |

Changes from the May 2025 version: typos, a list that LaTeX was silently
truncating (`%` in "90% of the mass"), and the quadratic-cost dual, which now
reads `(1/2) W_2^2` with the conjugate `psi*` on the source side, consistent with
`T* = grad psi*` used later in the text.

Build: `latexmk -pdf ot-foundations.tex` (needs `tcolorbox`, `physics`, `mathtools`, `microtype`).
