# Book Summary: *An Introduction to Information Theory*

- **File:** `C:\Users\Maha Mamdouh\Desktop\ML learning phase\Information theory introduction.pdf`
- **Sections Read:** Ch. 2.9–2.10 (Differential entropy), Ch. 3.1–3.4 (Data Processing Inequality, channel capacity, convex rate-distortion trade-off)

---

## 1. Key Mathematical Concepts Extracted

### 1.1 Differential Entropy and Gaussian Extremality (Ch. 2.9–2.10)
- For continuous random vector $X \in \mathbb{R}^d$, differential entropy is $h(X) = -\int p(x)\log p(x)\,dx$.
- Among all distributions with fixed covariance $\Sigma$, the multivariate Gaussian maximizes $h(X)$ with value $\frac{1}{2}\log\det(2\pi e \Sigma)$; adding independent Gaussian noise $\epsilon \sim \mathcal{N}(0, \sigma^2 I)$ adds noise power uniformly across all orthogonal modes.

### 1.2 Data Processing Inequality (DPI) & Rate-Distortion Theory (Ch. 3.1–3.4)
- **Data Processing Inequality:** For any Markov chain $X \to Y \to Z$, mutual information cannot increase: $I(X; Z) \le I(X; Y)$. Applying non-invertible spatial projections, downsampling, or low/high-pass frequency masks $\mathcal{P}_{\text{low}}, \mathcal{P}_{\text{high}}$ to a target variable $x$ strictly bounds reconstructible mutual information per branch.
- **Convex Rate-Distortion Function $R(D)$:** Lower bounds the minimum mutual information rate required to achieve expected MSE distortion $D$. Because high-frequency modes of natural signals have lower signal-to-noise ratio (SNR) under additive white Gaussian noise than low-frequency modes, achieving small distortion across high-frequency bands requires disproportionately high SNR or capacity; when SNR $\ll 1$, **no estimator (not even Bayes-optimal)** can achieve distortion below the noise-dominated floor (channel capacity constraint).

---

## 2. Direct Relevance to `HDMOEM_v3`
- **Spectral Partitioning via `laplacian_decomposition`:** Restricting `unet_loss` to $\mathcal{P}_{\text{low}}(D_{\text{unet}} - x)$ and `vit_loss` to $\mathcal{P}_{\text{high}}(D_{\text{vit}} - x)$ creates an asymmetry in SNR per frequency band under additive isotropic noise $n \sim \mathcal{N}(0, \sigma^2 I)$: most isotropic noise energy resides in the large high-frequency annulus ($\mathcal{P}_{\text{high}}$), whereas natural image/latent signal energy is concentrated in $\mathcal{P}_{\text{low}}$.
- **No-overfitting MSE floor:** A flat validation MSE with zero train-val gap reflects an irreducible distortion floor determined by optimizer equilibrium, model inductive bias, or Bayes risk under the noise schedule rather than data scarcity.
