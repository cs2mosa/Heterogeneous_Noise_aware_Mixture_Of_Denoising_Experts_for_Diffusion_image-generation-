# Phase 1 & 2 Source Registry

All primary mathematical books and scientific papers recursively discovered under `C:\Users\Maha Mamdouh\Desktop\ML learning phase\` and consulted for the mathematical diagnosis of `HDMOEM_v3`. Every citation in Phases 2–4 maps directly to one of the dedicated per-source summaries below.

---

## 1. Books & Lecture Texts (`analysis/sources/books/`)

| ID | Title & Author(s) | PDF Location | Pages / Sections Inspected | Dedicated Summary |
| :--- | :--- | :--- | :--- | :--- |
| `[B1]` | *Information Geometry and Its Applications* — Shun-ichi Amari | `deep learning math books\Information Geometry and Its Applications.pdf` | Ch. 4.2.6–4.2.8, Ch. 12.2.1–12.2.5, Ch. 13.6.2 | [amari_information_geometry.md](file:///f:/Heterogeneous-MOE-for-Diffusion-models/analysis/sources/books/amari_information_geometry.md) |
| `[B2]` | *Nonlinear Dynamics and Chaos* — Steven H. Strogatz | `deep learning math books\Nonlinear Dynamics and Chaos.pdf` | Ch. 2.4–2.7, Ch. 6.3 | [strogatz_nonlinear_dynamics.md](file:///f:/Heterogeneous-MOE-for-Diffusion-models/analysis/sources/books/strogatz_nonlinear_dynamics.md) |
| `[B3]` | *Deep Learning Architectures: A Mathematical Approach* — Ovidiu Calin | `deep learning math books\Deep-Learning-Architectures.pdf` | Ch. 4.5–4.8, Ch. 6.2.6–6.3 | [calin_deep_learning_architectures.md](file:///f:/Heterogeneous-MOE-for-Diffusion-models/analysis/sources/books/calin_deep_learning_architectures.md) |
| `[B4]` | *An Introduction to Information Theory* | `Information theory introduction.pdf` | Ch. 2.9–2.10, Ch. 3.1–3.4 | [information_theory_intro.md](file:///f:/Heterogeneous-MOE-for-Diffusion-models/analysis/sources/books/information_theory_intro.md) |
| `[B5]` | *The Math Behind Flow Matching and Diffusion Models* | `math behind flow matching and diffusion models.pdf` | Ch. 3.3, Ch. 4.1–4.3 | [diffusion_flow_matching_notes.md](file:///f:/Heterogeneous-MOE-for-Diffusion-models/analysis/sources/books/diffusion_flow_matching_notes.md) |
| `[B6]` | *Introduction to Riemannian Manifolds* — John M. Lee | `deep learning math books\Introduction to Riemannian Manifolds.pdf` | pp. 29–33, 71–74, 148–151 (Ex. 2.13, Prop. 2.16, Prop. 5.27) | [lee_riemannian_manifolds.md](file:///f:/Heterogeneous-MOE-for-Diffusion-models/analysis/sources/books/lee_riemannian_manifolds.md) |
| `[B7]` | *Image Processing Fundamentals* & *Probabilistic ML: Advanced Topics* (Murphy) | `image processing guide.pdf` (pp. 90–94); `Probabilistic ML advanced topics.pdf` (pp. 889–893) | Wiener filter (Eq. 194), Parseval/MSE, Laplacian edge mask (Eq. 192); DSM & Tweedie variance (Eq. 24.31–24.36) | [young_image_processing_and_murphy_dsm.md](file:///f:/Heterogeneous-MOE-for-Diffusion-models/analysis/sources/books/young_image_processing_and_murphy_dsm.md) |

---

## 2. Leading Research Papers (`analysis/sources/papers/`)

| ID | Title & Authors | PDF Location | Pages / Sections Inspected | Dedicated Summary |
| :--- | :--- | :--- | :--- | :--- |
| `[P1]` | *Analyzing and Improving the Training Dynamics of Diffusion Models (EDM2)* — Karras et al. (2024) | `scientifc papers\EDM_2 Analyzing and Improving the Training Dynamics of Diffusion Models.pdf` | Main pp. 1–9; App. B.1–B.7 & Config G pp. 24–31 | [edm2_karras_2024.md](file:///f:/Heterogeneous-MOE-for-Diffusion-models/analysis/sources/papers/edm2_karras_2024.md) |
| `[P2]` | *Elucidating the Design Space of Diffusion-Based Generative Models (EDM)* — Karras et al. (2022) | `scientifc papers\Elucidating the Design Space of Diffusion-Based Generative Models.pdf` | Sec. 5 (pp. 8–10); App. B.3, B.5–B.6 | [edm_karras_2022.md](file:///f:/Heterogeneous-MOE-for-Diffusion-models/analysis/sources/papers/edm_karras_2022.md) |
| `[P3]` | *nGPT: Normalized Transformer with Representation Learning on the Hypersphere* — Loshchilov et al. (2024) | `scientifc papers\NGPT (normalized transformers).pdf` | Main Sec. 2 (pp. 1–6); App. A.1–A.6 (pp. 10–15) | [ngpt_loshchilov_2024.md](file:///f:/Heterogeneous-MOE-for-Diffusion-models/analysis/sources/papers/ngpt_loshchilov_2024.md) |
| `[P4]` | *FreeU: Free Lunch in Diffusion U-Net* — Si et al. (2023) | `scientifc papers\FreeU, IMPORTANT TO READ.pdf` | Main Sec. 1–3.2 (pp. 1–6, Eq. 6–9) | [freeu_si_2023.md](file:///f:/Heterogeneous-MOE-for-Diffusion-models/analysis/sources/papers/freeu_si_2023.md) |
| `[P5]` | *DiffiT: Diffusion Vision Transformers for Image Generation* — Hatamizadeh et al. (2024) | `scientifc papers\DiffiT, Diffusion Vision Transformers for Image Generation.pdf` | Main Sec. 1–3.2 (TMSA) | [diffit_hatamizadeh_2024.md](file:///f:/Heterogeneous-MOE-for-Diffusion-models/analysis/sources/papers/diffit_hatamizadeh_2024.md) |
| `[P6]` | *DiffMoE: Dynamic Token Selection for Scalable Diffusion Transformers* — Shi et al. (2025) | `scientifc papers\DiffMoE.pdf` | Main Sec. 1–2.2 (batch-level token pool) | [diffmoe_shi_2025.md](file:///f:/Heterogeneous-MOE-for-Diffusion-models/analysis/sources/papers/diffmoe_shi_2025.md) |
| `[P7]` | *The Riemannian Geometry of Deep Generative Models* — Shao et al. (2017) | `scientifc papers\The Riemannian Geometry of Deep Generative Models.pdf` | Main Sec. 1–3 (pullback metric $G=J^\top J$) | [riemannian_geometry_generative_shao_2017.md](file:///f:/Heterogeneous-MOE-for-Diffusion-models/analysis/sources/papers/riemannian_geometry_generative_shao_2017.md) |
