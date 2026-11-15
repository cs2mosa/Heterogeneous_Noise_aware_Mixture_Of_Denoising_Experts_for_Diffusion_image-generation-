# Book Summary: *Introduction to Riemannian Manifolds*

- **Author:** John M. Lee
- **File:** `C:\Users\Maha Mamdouh\Desktop\ML learning phase\deep learning math books\Introduction to Riemannian Manifolds.pdf`
- **Pages Read:** pp. 29–33 (Riemannian submanifolds, Example 2.13, normal bundle, tangential/normal projections $\top, \perp$), pp. 71–74 (Euclidean spaces & Spheres with round metric $\mathring{g}_R$), pp. 148–151 (Proposition 5.27: Geodesics of spheres $S^n(R)$).

---

## 1. Key Mathematical Concepts Extracted

### 1.1 Riemannian Submanifolds, Normal Splitting & Orthogonal Projections (pp. 29–31)
- Every smooth embedded submanifold $M \subset \tilde{M}$ inherits the induced Riemannian metric $g = \iota^*\tilde{g}$ (Lemma 2.11, p. 29).
- **Round Metric on Spheres (Example 2.13, p. 30; pp. 72–73):** The $n$-sphere $S^n(R) \subset \mathbb{R}^{n+1}$ of radius $R$ carries the **round metric** $\mathring{g}_R$ induced by the Euclidean inner product on $\mathbb{R}^{n+1}$.
- **Orthogonal Direct Sum Splitting (pp. 30–31, Prop. 2.16):** At each point $p \in M$, the ambient tangent space splits orthogonally:
  $$T_p \tilde{M} = T_p M \oplus N_p M, \qquad N_p M = (T_p M)^\perp,$$
  with smooth bundle projections $\top: T\tilde{M}|_M \to TM$ and $\perp: T\tilde{M}|_M \to NM$.
- For the sphere $S^n(R)$, because $f(x) = \|x\|^2$ defines $S^n(R)$ with gradient $2p$, the tangent space is precisely $T_p S^n(R) = \{v \in \mathbb{R}^{n+1} : \langle v, p \rangle = 0\}$ (p. 149).

### 1.2 Geodesics on the Round Sphere $S^n(R)$ (Proposition 5.27, p. 149)
- A constant-speed curve on $S^n(R)$ is a geodesic if and only if its ambient acceleration $\gamma''(t)$ is everywhere normal to $T_{\gamma(t)} S^n(R)$ (i.e., proportional to $\gamma(t)$).
- **Explicit Geodesic Formula (p. 149):** Starting at $p \in S^n(R)$ with initial tangent velocity $v \in T_p S^n(R)$, setting $a = |v|/R$ and $\tilde{v} = v/a$ (so $|\tilde{v}| = R$):
  $$\gamma(t) = (\cos at)\, p + (\sin at)\, \tilde{v},$$
  traversing a great circle formed by intersecting $S^n(R)$ with $\operatorname{span}\{p, \tilde{v}\}$, maintaining $|\gamma(t)|^2 = R^2$ identically for all $t$.

---

## 2. Direct Relevance to `HDMOEM_v3`
- **`MP_Conv` & `normalize(x)` Tangent-Space Geometry:** Constraining weights $w \in S^{N-1}(\sqrt{N})$ restricts optimization to a round sphere of radius $R = \sqrt{N}$. The EDM2 forced-normalization gradient projection $(I - \hat{w}\hat{w}^\top)\nabla$ is literally Lee's tangential projection $\top$ onto $T_w S^{N-1}(\sqrt{N})$.
- **Testing `spherical_blend` against True Geodesic Interpolation on $S^n(R)$:** On a round sphere of fixed radius, geodesic (SLERP) interpolation between two vectors $u, v$ separated by angle $\theta$ preserves radius via $\frac{\sin((1-t)\theta)}{\sin\theta} u + \frac{\sin(t\theta)}{\sin\theta} v$ (a reparameterization of Prop. 5.27). Any blend formula that drops the $1/\sin\theta$ normalization factor, or mixes non-unit-norm vectors without re-normalization, **leaves the sphere manifold** and shrinks output variance when $\theta > 0$! This directly motivates a probe comparing `spherical_blend` norms vs. linear blending.
