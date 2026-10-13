# Important questions:

1) Can any Kriging model for 1D case produce the same curve (kriging mean) as Scipy smoothing splines? How to proove/disprove this numerically? Is there a mathematical confirmation?
2) Can any Kriging model for 1D case produce the same curve (kriging mean) as SMT's RMTB or RMTC? How to proove/disprove this numerically? Is there a mathematical confirmation?
3) Can any Kriging model for 1D case produce the same curve (kriging mean) as RMTS that has different mathematical formulation than those in SMT? How to proove/disprove this numerically? Is there a mathematical confirmation?
4) Can a Kriging model for 2D case produce the same surface (kriging mean) as SMT's RMTB or RMTC? How to proove/disprove this numerically? Is there a mathematical confirmation?
5) Since we confirmed that Kriging model for 1D case produces the same curve (kriging mean) as Scipy smoothing spline, can we evaluate variance (uncertainty) for the fitted spline curve without fitting kriging model?
6) Now sice we confirmed variance can be calculated independently for 1D-smoothing splines, can it be done for 2D splines? Is there a 2D smoothing spline support in Scipy?
7) How SciPy's make_smoothing_spline and/or UnivariateSmoothingSpline be extended to support varience estimation?
8) Does SciPy's RBFInterpolator with "thin_plate_spline" kernel produces B-spline surface for 2D case? Can knots, and control points be extracted from fitted model?
9) Is there direct N-D analogy of 'make_smoothing_spline' in SciPy or SMT, or any other python package? What is different between RMTS and make_smoothing_spline implementation? 
10) Explain iwp2_kernel + saddle_point_solve methodoloy. Why we compare it against smoothing spline? Is it gausian process or spline fitting?
11) Why our own matern32 implementation differs from SMT Kriging with "matern32" kernel? Are there extra hyperparameters we do not implement?
12) Can RBFInterpolator fitted model be converted to B-Spline or Nurbs representation directly?

---

## Answers (based on the Phase A/B/C work in this repo) 

### 1) Kriging vs. scipy smoothing splines

**Answer: yes, but not "any" Kriging model — a specific, named one.** A Universal
Kriging model with a **Matérn-(m−½) kernel**, polynomial trend of degree `m−1`, and
nugget scaled as `τ² = 4κ³λ` (cubic case, `m=2`) reproduces the scipy natural
smoothing spline of the matching degree **in the limit κ→0** (longer and longer
correlation length). A generic/arbitrary Kriging model — wrong kernel family (e.g.
squared-exponential), wrong trend degree, or a nugget not tied to `λ` this way — does
**not** converge to the spline; the identity holds for this one specific family, not
for Kriging in general.

**Mathematical confirmation:** this is exactly Corollary 5 of the paper (built on
Theorem 2a/2b's bordered-system identity and the kernel-reproducing-measure result
behind it, Proposition 4). It states that as the Matérn inverse length-scale `κ→0`
with the nugget held at `4κ³λ`, the Universal Kriging mean converges to the natural
smoothing spline of degree `2m−1` minimising `λ∫(f^(m))²dx` plus data fit.

**Numerical confirmation (reproducible, multiple independent datasets):**
- *Phase A* (`tests/test_phaseA_a3.py`, synthetic noisy `sin(6x)`): fitted convergence
  slope ≈ **1.2** on a log-log plot of `max|Kriging − spline|` vs. `κ` (asymptotic
  regime only — the full-range naive fit is biased to ≈1.6–1.7, a pitfall this
  repo hit and fixed). A separate, *exact* (non-asymptotic) check at `nugget=0` also
  confirmed the degree-1 (Matérn-½ ↔ linear spline) member of the same family.
- *`notebooks/smt_skills_cla_eta.ipynb`* (real, noiseless VLM `cla(η)` data), cell 22:
  with `λ=1e-2` matched exactly between the target spline and the nugget formula,
  `max|Kriging − spline|` falls monotonically from **5.343e-02 at κ=3** to
  **1.181e-04 at κ=0.01** (data range ≈ 0.67) — a clean, ~450× tightening.
- *`notebooks/test_hypotheses_2D_RANS.ipynb`*, Part A: the same κ-sweep repeated
  per-direction on real RANS CFD slices (both the `alpha` direction and the `mach`
  direction) gives the same clean monotone convergence, slopes in the 1.0–1.2 range,
  **once `λ` is matched** between the Kriging nugget and the target spline's own
  `λ` — using GCV's auto-selected `λ` for the target instead produces a spurious
  non-convergent plateau, a methodological trap this repo hit and documented
  (`sweep_1d` helper enforces matched `λ` now).
- Cross-check: a *generic*, maximum-likelihood-tuned `KRG(corr="matern32")` (no
  forced `κ→0`) lands at `max|scipy spline − Kriging| = 2.641e-02` on the same
  `cla(η)` grid — close, consistent with it implicitly sitting partway along the
  same convergence curve above, without being forced to the limit.

**How to disprove it, if false:** plot `max|Kriging − spline|` vs. `κ` on log-log
axes; the claim is falsified by anything other than a monotone decay with slope ≈1
in the asymptotic (small-`κ`) regime. This is exactly the test run above, and it
held every time it was tried, on both synthetic and two independent real datasets.

---

### 2) Kriging vs. SMT's RMTB or RMTC

**Answer: yes, but only with one very specific, non-generic kernel — not with a
generic stationary kernel such as plain Matérn.** Proposition 6 gives an *exact*
(not asymptotic) identity, but it requires building the Kriging kernel **directly
from RMTB/RMTC's own basis**: `k_m(x,x') = s²·φ(x)ᵀA⁻¹φ(x')`, where `φ` is RMTB/RMTC's
own basis-function vector and `A = H + βI` is its own energy/Hessian matrix plus the
regularization floor. A plain, generic Kriging model (any ordinary correlation
function, auto-tuned or forced to `κ→0`) does **not**, in general, reproduce RMTB/RMTC
closely — RMTB/RMTC are **finite-rank** objects (rank = number of control points),
while a generic Matérn kernel is full-rank; the two only coincide where the
finite-rank model happens to already be close to its own continuum (`κ→0`-spline)
limit.

**Mathematical confirmation:** Proposition 6 shows the finite-basis penalized
regression objective `J(w) = ½wᵀHw + 1/(2α)‖Φw−y‖²` used by RMTS-family models is
exactly `−log p(w|y)` for a Gaussian prior `w ~ N(0, s²A⁻¹)`, so its solution is the
posterior mean of a finite-rank GP with the kernel above. The equivalent dual/Woodbury
form is `ŵ = A⁻¹Φᵀ(ΦA⁻¹Φᵀ + αI)⁻¹y`.

**Numerical confirmation:**
- *`smt_skills_cla_eta.ipynb`*, cells 24–27 (cubic `cla(η)` data, `RMTB`,
  `num_ctrl_pts=12`): the **dual identity itself** (weight-space vs. function-space
  form of the same `A`, `Φ`) is confirmed to **1.454e-08** once conditioning is
  handled (`β=1e-6` instead of SMT's own `β=1e-14`, which is too ill-conditioned for
  float64 — `cond(A)≈1.55e13` there). This is Proposition 6 holding exactly, as pure
  algebra, independent of SMT's solver.
- **But** that reconstruction differs from SMT's *actual* `predict_values` output by
  **4.775e-02 (RMTB)** and **3.512e-02 (RMTC)**, flat across `β` from `1e-14` to
  `1e-6` — i.e. not a conditioning artifact. SMT's real Krylov solver is minimizing
  something close to, but subtly different from, the textbook `J(w)` built from its
  own exposed `full_hess`/`full_jac_dict` matrices. This gap is **unresolved** (logged
  in the Research Plan, Phase C3) and is the main caveat on "exact" above.
- *`test_hypotheses_2D_RANS.ipynb`*, cell 32 (real 2-D RANS data, `m=400≫n=35`): the
  same reconstruction-vs-SMT gap is **much tighter at and near training points**
  (`~2e-9` CD, `~1e-8` CL, again `β`-independent) but **grows to `~2e-2` in
  sparse-coverage regions**, and *there* it **is** `β`-sensitive — a real
  null-space/conditioning effect (Remark 7's "vague proper prior"), distinct from the
  flat training-point gap. Solving this system required the dual/Woodbury route as a
  numerical *necessity*: the naive primal 400×400 system has `cond ≈ 6e22` and returns
  garbage.
- Contrast with a *generic* (non-exact-kernel) Kriging model: on the same `cla(η)`
  grid, a plain auto-tuned `KRG(corr="matern32")` gives `max|RMTB − Kriging| =
  6.543e-02` and `max|RMTC − Kriging| = 6.270e-02` — markedly worse than its
  `2.641e-02` gap to the scipy spline (Q1). This is the expected signature of
  RMTB/RMTC being finite-rank, not full-rank: a generic stationary kernel gets
  noticeably closer to the continuum spline than to the finite-rank RMTB/RMTC curve
  built with the library-default `energy_weight=1e-4`.

**How to disprove it, if false:** build `k_m` from the model's own `full_hess` /
`full_jac_dict` (and, for `RMTC`, map through `full_dof2coeff` first) and check the
weight-space vs. function-space forms agree; the claim is disproved by a persistent,
`β`-independent disagreement — which is in fact what happened one level up, not in
the dual identity itself but between the identity and SMT's own solved output. That
distinction (exact math vs. imperfect real solver) is the honest, numerically
demonstrated answer here, not a clean "yes" or "no."

---

### 3) Kriging vs. an "RMTS-like" model with a different formulation

**Answer: yes — this is the more general and more interesting result.** Proposition
6's identity is **not** a quirk of SMT's specific B-spline implementation. It is a
completely general statement about *any* finite-basis penalized-regression objective
of the form `J(w) = ½wᵀHw + 1/(2α)‖Φw−y‖²`, for **any** choice of basis `Φ` and
quadratic energy matrix `H` — B-splines, Hermite finite elements, or any other finite
basis someone might use to build an "RMTS with a different formulation." Whatever that
basis is, a Kriging model using *that basis's own* finite-rank kernel `k_m` will match
it exactly, by the same proof.

**Mathematical confirmation:** Proposition 6's proof never references B-splines
specifically — `Φ` and `H` are generic matrices. The identity is basis-agnostic by
construction.

**Numerical confirmation that this generality is real, not assumed:** this repo
tested Proposition 6 on **two different bases with two different `Φ`/`H` structures**
and got the same result both times:
- `RMTB` (B-spline basis): dual identity exact to `1.454e-08`.
- `RMTC` (cubic Hermite finite elements — a structurally different basis, with an
  extra dof→coefficient map `full_dof2coeff` that `RMTB` doesn't have): dual identity
  exact to the same order, and the gap to its own `predict_values` output
  (`3.512e-02`) is the same order of magnitude as `RMTB`'s (`4.775e-02`) — pointing to
  a shared `RMTS`-base-class solver detail, not a B-spline-specific one (see Q2).

This is direct evidence that the Kriging-equals-RMTS-mean identity is a property of
*the class of finite-basis penalized models*, not of SMT's particular choice of basis
— so yes, a hypothetical "RMTS with a different mathematical formulation" would be
matched by Kriging the same way, provided the matching finite-rank kernel is built
from *that* formulation's own `Φ` and `H`.

**The caveat carries over unchanged from Q2:** the guarantee is for the *textbook*
objective a formulation is defined by, not necessarily for a specific software's
*actual* numerical output. SMT's own RMTB and RMTC solvers each deviate a few percent
from their own textbook objective (Q2) for reasons not yet root-caused in this repo.
A different RMTS-like implementation could plausibly have the same kind of
solver-vs-objective gap, for its own implementation-specific reasons (different
quadrature, different convergence tolerances, different boundary handling) — that
part is a property of *software*, not of the *mathematics*, and would need to be
checked per-implementation the same way it was checked here for RMTB/RMTC.

**What was not yet tested (an honest gap, not a claim):** the Research Plan's Phase
B1 — a fully from-scratch, SMT-independent finite-basis implementation (e.g. a
hand-rolled B-spline or radial basis set with its own `Φ`/`H`, with no SMT code in
the loop at all) — was planned but never built. The RMTC cross-check above is strong
supporting evidence for basis-independence, but a from-scratch non-SMT example would
make the "any formulation" claim airtight rather than inferred from two SMT variants.
Also worth noting as a contrast: `LoftedSmoothingSpline` (a from-scratch 2-D model
built by lofting 1-D `scipy` smoothing splines across Mach) is a genuinely
**non-variational** construction, not of the `J(w)` form at all — Proposition 6
does **not** apply to it, which is itself a useful boundary case for "any
formulation": the identity covers any penalized-regression formulation, but not
every possible way of building a surrogate model.

**How to disprove it, if false:** take any other finite-basis penalized model,
extract its own `Φ`/`H` (or equivalent), build `k_m` from them, and check the dual
identity. It would be disproved by a persistent mismatch that — unlike the RMTB/RMTC
solver gap — survives even when compared against *that same model's own* textbook
objective rather than its solved output.

---

### 4) Kriging vs. SMT's RMTB or RMTC — the 2-D case

**Answer: the same split answer as Q2, but now checked directly on real 2-D data**
(`data/rans_data.csv`: `C_d`, `C_l` vs. angle of attack `α` and Mach `M`), where the
two routes to "yes" (Proposition 6 vs. Corollary 5) diverge sharply instead of both
being plausible.

**Via Proposition 6 (finite-rank construction, exact): yes, essentially exactly.**
Nothing in the proof is 1-D-specific — `Φ` and `H` are just matrices, regardless of
how many input dimensions `x` has — so the same dual/function-space identity used in
Q2 applies unchanged to a real `RMTB` model with 2-D inputs, `m=400` control points
fit to `n=35` training points.

- *`notebooks/test_hypotheses_2D_RANS.ipynb`*, cell 32: the dual identity itself
  agrees with the reconstructed function-space form to `~1e-15` everywhere, by
  construction. The gap to SMT's *actual* `RMTB` output is **`~2e-9` (CD) / `~1e-8`
  (CL)** at and near training points — `β`-independent, and *much tighter* than the
  1-D `cla(η)` case's `~5%` gap (Q2) — but **grows to `~2e-2` in sparse-coverage
  regions**, and there it *is* `β`-sensitive (a genuine conditioning effect in the
  energy-governed null space, not the same phenomenon as the flat training-point
  gap).
- Here the dual/Woodbury route isn't just elegant, it's **numerically mandatory**:
  the naive primal 400×400 system has `cond(A) ≈ 6e22` in float64 and returns
  garbage; only the dual form (inverting the `n×n` kernel Gram matrix instead of the
  `m×m` Hessian) is solvable at all.
- So: *why* is the 2-D training-point gap (`~2e-9`) so much tighter than the 1-D
  gap (`~5%`, Q2)? Not yet root-caused — logged as open in the Research Plan
  (Phase C3) — plausibly related to `m≫n` here vs. a more balanced `m`/`n` ratio in
  the 1-D case, or to the specific `num_ctrl_pts`/`energy_weight` choice, but this
  is a hypothesis, not a confirmed explanation.

**Via Corollary 5 (full-rank asymptotic limit): not demonstrated in 2-D.** This is
the part where the 2-D case genuinely differs from — and is weaker than — the 1-D
story in Q1/Q2.

- Part A of the same notebook (cells 33–39) re-confirmed the 1-D building blocks
  cleanly on real slices of this exact dataset: the per-direction `α`-slice and
  `M`-direction sweeps both converge with the matched-`λ` nugget formula, the same
  clean monotone behavior as Q1.
- Part B (cells 40–45) then tried the natural next step — a genuine 2-D
  tensor-product Matérn-3/2 kernel, one shared decay-rate parameter `c`, nugget
  `=4c³λ` — to see if it converges toward `RMTB`'s surface as `c→0`, the way the
  paper's Table 1 footnote gestures at ("tensor-product energies with one weight per
  direction") without actually deriving. **It does not converge**: error vs. `c` is
  roughly flat (even increasing at first), and a direct nugget search at small `c`
  only reaches `~50–80%` of CD's data range as its best achievable match — nowhere
  near Proposition 6's `~2e-9` training-point agreement (not a fully fair comparison,
  since one is an exact identity and the other an asymptotic limit, but the contrast
  is stark).
- Part C (cells 46–48) ruled out the obvious confound — a `λ`/`energy_weight`
  mismatch, the same bug class that caused Q1's mach-direction false plateau — by
  re-running with `λ` matched to `RMTB`'s actual `energy_weight`. **Still no
  convergence.** This sharpens rather than resolves the finding: what breaks is
  specifically the *combination* into one shared-parameter tensor kernel, likely
  because a true tensor-product kernel's RKHS norm needs per-direction weights and
  cross terms that `RMTS`'s own *additive* energy (`H = Σ_l s_l H_l`) doesn't need,
  and that one shared `c` can't represent. The paper never works out the correct
  multi-D nugget-scaling law, so there is currently no known-correct target to test
  against — this is an **open problem**, not a disproof of Corollary 5 itself.

**Mathematical confirmation:** Proposition 6 (dimension-agnostic by construction, as
in Q2/Q3) — confirmed. Corollary 5's extension to more than one dimension is
**not** derived in the paper (Table 1 only gestures at it) and this repo's own
attempt to derive and test a tensor-product version did not succeed — so there is
currently no mathematical confirmation for the 2-D asymptotic route, only a
confirmed negative result for one reasonable heuristic.

**How to prove/disprove numerically:** for Proposition 6, extract the trained
model's `full_hess`/`full_jac_dict` (map through `full_dof2coeff` for `RMTC`) exactly
as in Q2, regardless of input dimension, and check the dual identity plus the gap to
`predict_values` — both at training points and away from them, since the 2-D case
showed those two behave differently. For Corollary 5, run a κ/`c`-sweep of a
tensor-product Kriging model against the `RMTB`/`RMTC` surface with matched `λ`;
the claim is disproved by a non-monotone or flat error curve as `c→0` — which is
exactly what happened here.

### 5) Can we get the fitted spline's uncertainty without fitting a Kriging model?

**Answer: yes, in principle and in closed form — but this repo has not implemented
or numerically tested it, so treat this as a derived, theoretically-grounded answer,
not a confirmed one.** Corollary 5 (Q1) already establishes that the spline *is* the
mean of a specific Gaussian process — a Matérn-$(m-\tfrac12)$ Universal Kriging model
in the $\kappa\to0$, `nugget`$=4\kappa^3\lambda$ limit. If a mean has a GP
interpretation, that same GP has a variance, by definition — the real question is
whether getting it requires the expensive, generic part of "fitting a Kriging model"
(choosing a correlation family, estimating/sweeping a length-scale hyperparameter) or
whether it collapses to something cheaper once the limit is taken.

**Mathematical confirmation.** Theorem 2b already gives the general Universal-Kriging
predictive-variance formula for *any* kernel/trend pair satisfying the paper's
assumptions — this repo's `krig_splines/uk.py` implements its mean (`gls_predict`,
`saddle_point_predict`) but **no variance function exists in this codebase**
(checked: no `def` matching `variance`/`var` anywhere in the package). Corollary 5
specifies *which* kernel has a mean that reduces to the spline; the correct
variance is, by construction, whatever Theorem 2b's variance formula gives for that
same kernel/nugget prescription, in the same $\kappa\to0$ limit. Two things make this
a non-trivial limit, not a free lunch:
- The raw Matérn-$(m-\tfrac12)$ covariance **degenerates** as $\kappa\to0$ — at
  $\kappa=0$, $k(x,x')=1$ for all $x,x'$ (rank-1, a shared offset, not an
  interesting correlation structure). The mean limit is only well-defined because
  the nugget is driven to zero at the *same* rate ($4\kappa^3\lambda$) — a joint
  limit, not a limit of the kernel alone.
- The classical resolution of exactly this joint limit predates the paper: it is
  the **Kimeldorf–Wahba correspondence** (Kimeldorf & Wahba, 1970) and Wahba's
  "Bayesian confidence intervals for smoothing splines" (Wahba, 1983; *Spline
  Models for Observational Data*, 1990). There, the smoothing spline is derived
  directly as the posterior mean of a *partial* Bayesian model — a diffuse
  (improper, infinite-variance) prior on the degree-$(m-1)$ polynomial part `P`
  (exactly this repo's trend basis), plus a proper Gaussian process prior with a
  known, closed-form reproducing kernel $G(x,x')$ on the "rough" part, where $G$ is
  the Green's function of the penalty operator (e.g. for the cubic case, $m=2$, $G$
  is a specific piecewise-cubic polynomial — not the Matérn family at all, since the
  Matérn parametrization has already been eliminated by the limit). That
  construction gives $\mathrm{Var}[f(x^*)\mid y]$ directly, via Theorem 2b's same
  formula applied once, with $G$ in place of the Matérn kernel and no $\kappa$ to
  sweep or fit.

**What this means operationally — "without fitting a Kriging model" is accurate in
the sense that matters:** the expensive part of "fitting a Kriging model" is
estimating a length-scale/correlation hyperparameter (maximum likelihood, a $\kappa$
sweep, cross-validation). None of that is needed here. Once $m$ (the spline's
derivative order) and $\lambda$ (already fixed — the same $\lambda$ passed to
`make_smoothing_spline`) are known, $G(x,x')$ is a *known, closed-form* function with
no free parameters, and Theorem 2a's bordered system with $G$ in place of $K$ is a
single, well-posed linear solve (no near-degenerate kernel, since $G$ does not
collapse to a constant the way the raw Matérn kernel does at $\kappa=0$) — cheaper
and more numerically direct than the $\kappa\to0$ Kriging route used throughout this
notebook/document, not an equivalent amount of work in disguise.

**A second, even cheaper, classical shortcut — but training-points only.** Wahba's
same treatment gives a simpler formula *at the training points*: if $S_\lambda$ is
the smoothing spline's own linear smoother ("hat") matrix (defined by
$\hat y = S_\lambda y$ — the same matrix `make_smoothing_spline`'s fitting procedure
applies, just not exposed by `scipy`'s API), then
$\mathrm{Var}[\hat f(x_i)] \approx \hat\sigma^2\,[S_\lambda S_\lambda^\top]_{ii}$,
with $\hat\sigma^2$ estimated from the fit residuals. This needs no kernel, no
Green's function, and no Kriging machinery at all — just the smoother matrix the
spline fit already implicitly constructs. It does not directly give variance at an
arbitrary *query* point between data points, though; for that, the $G(x,x')$-based GP
route above is the general answer.

**Status in this repo: implemented and numerically confirmed** —
`notebooks/spline_variance_without_kriging.ipynb`, backed by
`krig_splines.kernels.iwp2_kernel` and `krig_splines.uk.saddle_point_variance`
(new package code, covered by `tests/test_spline_variance.py`). Two results, on the
same synthetic noisy-$\sin(6x)$ benchmark Phase A uses:

- **Mean:** the closed-form $G$-kernel, `nugget=lambda` directly, one linear solve,
  reproduces `scipy`'s smoothing spline to `~1e-12` — tighter than Q1's $\kappa\to0$
  sweep (`~1e-4` at its smallest tested $\kappa$), because there is no limit to
  approximate.
- **Variance — and a real finding, not just a confirmation:** the *naive* version of
  the check (sweep $\kappa\to0$, compare `saddle_point_variance` under the Matérn
  kernel to the $G$-kernel's variance) **does not converge** — the raw Matérn
  variance collapses to zero as $\kappa^3$ instead of reaching a finite limit. The
  shape is already exactly right at every $\kappa$ (correlation `1.0000` against the
  $G$-based variance); only the *scale* is wrong, by precisely $4\kappa^3$ — the same
  constant the mean's `nugget`$=4\kappa^3\lambda$ formula already uses, now shown to
  apply to the kernel's own scale too, not only the nugget. Dividing the raw Matérn
  variance by that same $4\kappa^3$ before comparing gives clean, monotone
  convergence to within `0.02%` of the $G$-based variance's own range by
  $\kappa=0.003$.

This sharpens the original answer: getting the variance from the $\kappa\to0$ Kriging
route is possible, but is *not* a free corollary of the mean's recipe the way it might
look at first — it requires separately discovering that the kernel's absolute scale,
not just the nugget-to-kernel ratio, needs the same rescaling. The closed-form
$G$-kernel route sidesteps this entirely (no $\kappa$, no rescaling to find), which is
the practical reason it is the better answer to "without fitting a Kriging model,"
not merely an equivalent one.

**What remains untested:** only $m=2$ (cubic) was implemented; general $m$ has an
analogous closed form (Kimeldorf & Wahba, 1970) but was not coded. The real
`cla(eta)`/RANS datasets used elsewhere in this document were not re-run through this
closed-form route — only the synthetic benchmark was.

---

### 6) Does the same variance trick work in 2-D? Does scipy have 2-D smoothing splines?

**Answer: yes to both — but "yes" to the 2-D variance question answers a different
question than Q4's "does Kriging converge to RMTB" one, and the two must not be
conflated.**

**Does scipy have 2-D smoothing-spline support?** Yes, in two genuinely different
forms:
- `scipy.interpolate.SmoothBivariateSpline` / `bisplrep` / `RectBivariateSpline` —
  FITPACK's (Dierckx) tensor-product B-spline smoother, controlled by a residual-sum
  constraint `s`, not a `λ`-penalized derivative norm. A different variational
  criterion from everything else in this document.
- **`scipy.interpolate.RBFInterpolator(kernel="thin_plate_spline", smoothing=lam)`**
  — this one *is* the direct, classical 2-D generalization of
  `make_smoothing_spline`, and it is closer to this project's own framework than
  its name suggests: scipy's own docstring for `RBFInterpolator` writes its solve as

  $$\big(K(x,y)+\lambda I\big)a + P(y)\,b = d, \qquad P(y)^\top a = 0$$

  — **literally Theorem 2a's bordered system**, down to calling the smoothing
  parameter $\lambda$, and cites Wahba (1990) directly in its references. `degree=1`
  (the default, and the minimum scipy allows for this kernel) is exactly the
  degree-1 polynomial null space $P=\mathrm{span}\{1,x,y\}$ this document has used
  throughout, one dimension up. The kernel itself, `r**2 * log(r)`, is Duchon's
  (1977) reproducing kernel for the 2-D bending-energy penalty
  $\iint\big[(f_{xx})^2+2(f_{xy})^2+(f_{yy})^2\big]\,dx\,dy$ — the direct 2-D
  analogue of `iwp2_kernel`'s $\int(f'')^2dx$.

**Mathematical confirmation:** nothing in Theorem 2a/2b or `saddle_point_solve`/
`saddle_point_variance` assumes 1-D input — both already operate on generic `K`,`P`
matrices. Duchon's classical multi-D extension of the Kimeldorf–Wahba theory is the
textbook justification for the thin-plate-spline kernel playing the same role in 2-D
that `iwp2_kernel` plays in 1-D.

**Numerical confirmation (new, run for this answer, not yet in a notebook):** on
synthetic 2-D data ($n=30$, $\mathrm{sin}(3x)\cos(3y)$ plus noise),
`saddle_point_solve` with `K(x,x')=\lVert x-x'\rVert^2\log\lVert x-x'\rVert`,
`P=[1,x,y]`, `nugget=lambda` reproduces
`RBFInterpolator(kernel="thin_plate_spline", smoothing=lambda, degree=1)`'s output
to **6.2e-15** (data range 1.75) — the same order of agreement Q1 found in 1-D.
`saddle_point_variance`, same `K`/`P`, unmodified code, runs with no changes and
gives well-behaved results: variance roughly **15× larger** in sparsely-sampled/
extrapolated corners than inside the densely-sampled region (0.329 vs. 0.022 on this
test), the expected qualitative Kriging-variance shape.

**The caveat that matters:** this is a *different* 2-D generalization from the one
`test_hypotheses_2D_RANS.ipynb` Part B/C went looking for (Q4). That notebook tried
a *tensor-product* kernel — separate length scales per direction, combined
multiplicatively — to approximate `RMTB`'s own *additive* per-direction energy
($H=\sum_l s_l H_l$), and found it did not converge. The thin-plate spline tested
here is **isotropic** (one radial kernel, no per-direction structure at all) and
was never claimed to match `RMTB`'s specific construction — it answers "does the
Theorem 2/Kimeldorf–Wahba variance machinery generalize to 2-D smoothing splines,"
which it does, cleanly, but this does **not** resolve Q4's still-open question about
Kriging converging to `RMTB`'s own 2-D surface. Tensor-product energy, additive
RMTS-style energy, and isotropic thin-plate bending energy are three distinct
mathematical objects; confirming one says nothing about the other two.

**Status in this repo:** the recipe above is verified (ad hoc, for this answer) but
**not yet added** to `krig_splines` as a `tps2d_kernel` function or exercised in a
notebook — a natural, low-effort follow-up (no changes needed to
`saddle_point_solve`/`saddle_point_variance` themselves, only a new kernel function
parallel to `iwp2_kernel`), not yet done.

---

### 7) How could `make_smoothing_spline` (or `UnivariateSpline`) be extended to support variance?

*(Note: scipy has no class named `UnivariateSmoothingSpline` — there is
`scipy.interpolate.UnivariateSpline`, a different, older, FITPACK-based smoothing
spline, and `scipy.interpolate.make_smoothing_spline`, the Reinsch/Weinert natural
cubic smoothing spline used throughout this document. They are answered separately
below since they do not share an internal parametrization.)*

**Answer: don't patch scipy — wrap it.** The lowest-effort, most robust extension is
a thin companion class that calls `make_smoothing_spline` unchanged for the mean
(so it is guaranteed bit-identical to scipy's own answer) and separately computes
`saddle_point_variance` with `iwp2_kernel` for the variance, reusing the exact
recipe Q5/Q6 already proved correct:

```python
class SmoothingSplineWithVariance:
    def __init__(self, x, y, lam):
        self.x, self.lam = np.asarray(x), lam
        self._spline = make_smoothing_spline(x, y, lam=lam)   # scipy, untouched
        self._K = iwp2_kernel(self.x[:, None], self.x[None, :])
        self._P = poly_basis(self.x, degree=1)
        self._a, self._beta = saddle_point_solve(self._K, self._P, y, nugget=lam)

    def mean(self, xq):
        return self._spline(xq)                                # == scipy exactly

    def variance(self, xq):
        xq = np.asarray(xq)
        Kq = iwp2_kernel(xq[:, None], self.x[None, :])
        Pq = poly_basis(xq, degree=1)
        Kqq = iwp2_kernel(xq, xq)
        return saddle_point_variance(self._K, self._P, self.lam, Kq, Pq, Kqq)
```

**Why not modify `make_smoothing_spline` itself:** checked directly — it returns a
bare `scipy.interpolate.BSpline`, exposing only `t`, `c`, `k` (knots, coefficients,
degree). The design matrix, `λ`, and the original `x`/`y` are computed internally
and discarded; there is nothing to "extend" after the fact without either re-running
the fit (what the wrapper above does) or patching scipy's own C/Fortran-backed
fitting routine upstream. The wrapper sidesteps this entirely and depends on zero
scipy internals, only its public, documented output.

**`UnivariateSpline` is a real complication, not just a rename.** It is FITPACK's
(Dierckx) algorithm, parametrized by a residual-sum target `s`, not `λ` directly;
`s` and `λ` both control the same underlying bias–variance trade-off but are not the
same number, and FITPACK's internal `s`-to-effective-`λ` mapping is not exposed by
scipy's public API. Attaching the same variance recipe to a `UnivariateSpline` fit
would need that effective `λ` recovered first — e.g. by root-finding `λ` in
`make_smoothing_spline(x, y, lam=λ)` until its curve matches the `UnivariateSpline`
fit to within solver tolerance — a solvable but **real extra step, not yet
attempted or tested here**, and a genuine caveat on how directly this answer
transfers from `make_smoothing_spline` to `UnivariateSpline`.

**Extends to 2-D for free.** The same wrapper pattern applies unchanged to
`RBFInterpolator(kernel="thin_plate_spline", smoothing=lam)` using Q6's just-verified
2-D kernel in place of `iwp2_kernel` — one uniform pattern (call scipy for the mean,
call `saddle_point_variance` with the matching closed-form kernel for the variance)
covers both the 1-D and 2-D cases documented here.

**Status in this repo:** sketched and spot-checked above (the mean/variance pieces
it calls are implemented and tested — `iwp2_kernel`, `saddle_point_variance`), but
the wrapper class itself is not committed to `krig_splines` as of this answer.

---

### 8) Is `RBFInterpolator`'s thin-plate-spline surface a B-spline? Can knots/control points be extracted?

**Answer: no, and no.** `RBFInterpolator(kernel="thin_plate_spline")` does **not**
produce a B-spline surface, and consequently has no knots or control points to
extract — it is a different kind of object entirely, verified directly below rather
than assumed.

**What it actually is.** A fitted `RBFInterpolator` evaluates as

$$f(x) = \sum_{i=1}^n a_i\,K(x, y_i) + P(x)^\top b, \qquad K(x,y)=\lVert x-y\rVert^2\log\lVert x-y\rVert$$

— a sum of radial kernels **centered at the original training-data locations**
$y_i$, plus the degree-1 polynomial trend, exactly Theorem 2a's bordered-system
predictor (confirmed in Q6). A B-spline surface, by contrast, is

$$f(x,y) = \sum_i\sum_j c_{ij}\, B_i(x)\, B_j(y)$$

— a tensor product of **piecewise-polynomial basis functions defined on a knot
vector**, with coefficients $c_{ij}$ ("control points") that sit at positions
*implied by the knot spacing*, generally **not** at the data points at all. These
are different mathematical objects with different structure, not two names for the
same thing:

| | `RBFInterpolator` (thin-plate) | `SmoothBivariateSpline` (real B-spline) |
|---|---|---|
| Basis support | **global** — every $K(x,y_i)$ is nonzero everywhere except exactly at $y_i$ | **local/compact** — each $B_i$ is nonzero only on a few knot intervals |
| "Node" locations | the $n$ training points themselves | knots, placed by FITPACK's own algorithm, generally $\neq$ data points |
| Coefficients | $n$ kernel weights + a small polynomial trend | control points, one per tensor-product basis pair |
| Changing one coefficient | perturbs the **whole surface** (dense coupling) | perturbs only a **local patch** (compact support) |

**Verified directly, not assumed** (checked both objects' real attributes in this
session):
- `RBFInterpolator` exposes `.y` (the $(n,2)$ training-point locations — these are
  the kernel "centers," not knots), `.d` (training values), `._coeffs` (the
  combined $[a;b]$ weight vector, $n+q$ entries for degree-1's $q=3$ polynomial
  terms), and `.powers` (the monomial exponents defining $P$ — `[[0,0],[1,0],[0,1]]`
  for degree 1, i.e. $\{1,x,y\}$). **No knot vector exists anywhere on the object**
  — `dir(rbf)` has nothing named `t`, `tx`, `ty`, or `knots`.
- `SmoothBivariateSpline`, by contrast, genuinely has them: `.get_knots()` returns
  real `(tx, ty)` knot vectors, `.get_coeffs()` returns the real tensor-product
  control-point array (length `(len(tx)-k-1)*(len(ty)-k-1)`), and `.degrees` gives
  `(kx, ky)`. This is the object that actually has what Q8 is asking for — just not
  the one Q6 used for the variance identity (Part 5 of
  `notebooks/spline_variance_2d_vs_smt.ipynb` covers it, with no variance route,
  per Q6/Q7's caveat).

**One caveat found while checking this, worth being upfront about rather than
glossing over:** `RBFInterpolator._coeffs` is a **private, underscore-prefixed**
attribute — not documented, not guaranteed stable across scipy versions. It also
turns out *not* to equal `krig_splines`'s own `[a; beta]` from `saddle_point_solve`
directly (`max diff ≈ 0.51` on a quick check, not a rounding-level gap) — scipy
internally shifts/scales the input coordinates first (`._shift`, `._scale`
attributes, found when investigating the mismatch), the same kind of
coordinate-normalization wrinkle Part 4 of `spline_variance_without_kriging.ipynb`
had to reverse-engineer carefully for SMT's `KRG`. That reconciliation was **not**
carried out here — flagged as unverified rather than claimed, since it is tangential
to this question's actual answer (no knots exist either way) and would need the same
careful, dedicated treatment the SMT case got, not a quick guess.

**Practical takeaway:** if you specifically need a knot/control-point B-spline
representation of a smoothed 2-D surface, use `SmoothBivariateSpline`/`bisplrep`
(Part 5), not `RBFInterpolator`. If you need the exact Theorem 2a/Kimeldorf–Wahba
variance machinery this document builds on, use `RBFInterpolator`/`SmoothingSpline2D`
(Q6/Q7), not `SmoothBivariateSpline`. No scipy tool explored in this document gives
both at once — they are genuinely different representations of "a smoothed surface,"
not interchangeable implementations of the same idea.

---

### 9) Is there a direct N-D analogue of `make_smoothing_spline`? What actually differs between RMTS and `make_smoothing_spline`?

**Part 1: a direct N-D analogue.** Yes — but "direct" splits into two genuinely
different answers depending on which *algorithm family* you mean by "analogue," a
distinction this document has already drawn (Q6/Q8) and that matters here too:

- **`scipy.interpolate.RBFInterpolator(kernel="thin_plate_spline")`** is the
  mathematically exact N-D generalization of the *same variational principle*
  `make_smoothing_spline` solves (Duchon 1977's extension of Kimeldorf–Wahba to
  $d$ dimensions) — and it is **not limited to 2-D**: its own docstring states
  "Radial basis function interpolator in **N ≥ 1 dimensions**," confirmed already
  in this document for $N=1,2$ (Q5/Q6). This is the same `λ`-penalized bending-energy
  framework, same bordered system, same null-space-plus-kernel structure, just with
  a different (global radial, not banded-1-D) solver underneath.
- **`csaps`** (a separate PyPI package, **not** scipy or SMT — confirmed installed
  and exercised directly for this answer, version `1.3.3`) is the more literal
  algorithmic analogue: `csaps.NdGridCubicSmoothingSpline` is a genuine N-D
  **tensor-product natural cubic smoothing spline**, the same algorithm family
  `make_smoothing_spline` uses, extended dimension-by-dimension — confirmed by
  running it (`NdGridCubicSmoothingSpline([gx, gy], Z, smooth=[0.9, 0.9])` fits and
  evaluates correctly). Two things worth noting precisely, not glossed over:
  **it requires data on a rectangular grid** (`xdata` is a *sequence* of 1-D
  per-axis coordinate vectors, not scattered points — the same practical constraint
  Part 5 found for `SmoothBivariateSpline`), and **it takes a separate smoothing
  parameter per dimension** (`smooth=[0.9, 0.9]` above, confirmed via
  `sp.smooth → (0.9, 0.9)`) — i.e. it is structurally an **additive/tensor-product**
  construction, not an isotropic one. That makes it the one package in this answer
  closer in *mathematical structure* to RMTS's own additive energy
  ($H=\sum_l s_l H_l$) than to `RBFInterpolator`'s isotropic thin-plate energy — a
  genuinely useful distinction for Part 2 below.
- **scipy has no N-D version of `SmoothBivariateSpline`/`bisplrep`** — FITPACK's
  `surfit` is bivariate only; there is no trivariate-or-beyond FITPACK smoothing
  spline in scipy.
- Not verified in this session (general knowledge only, flagged rather than
  asserted as checked): R's `fields::Tps` is the same thin-plate-spline
  construction as `RBFInterpolator` in a different language; general-purpose GP
  packages (`GPy`, `GPflow`, `sklearn.gaussian_process`) could reproduce this
  *in principle* via Corollary 5's own kernel construction, but none ship it as a
  ready-made "smoothing spline" function — you would be hand-building the same
  bordered system this repo's `krig_splines` already does, not calling a library
  feature.

**Part 2: RMTS vs. `make_smoothing_spline` — what is actually different.**
Pulling together what Q1–Q8 already established into one direct comparison, rather
than re-deriving it:

| | `make_smoothing_spline` | RMTS (`RMTB`/`RMTC`) |
|---|---|---|
| Basis | **exact** — natural cubic spline, knots *at the data points*, rank $=n$ | **finite, user-chosen** — B-spline/Hermite basis on a separate grid over `xlimits`, rank $=$ `num_ctrl_pts` (independent of $n$; Q2's `m≫n` regime) |
| Energy term | **exact** $\int (f'')^2\,dx$ | **quadrature approximation** of it, exactness controlled by `approx_order` — shown in this repo to matter a great deal (`approx_order=2` vs. the library default `4` gave a `2.6%` vs. `17.75%` gap to the target spline on the same slice, Q4's `num_ctrl_pts` follow-up) |
| Solver | **direct, exact** banded linear algebra (de Boor/Reinsch/Weinert), one solve, no iteration | **iterative nonlinear solver** (Newton steps, `nonlinear_maxiter`/`line_search`) — confirmed in this repo to *not fully converge* at extreme `num_ctrl_pts`/data ratios (Q4's follow-up: error blew up to `149%` of range at `num_ctrl_pts=400`, `n=7`, only partially recovered by raising `nonlinear_maxiter`) |
| Smoothness control | **one** scalar `lambda`, with a well-defined statistical meaning (GCV-computable) | **several, coupled** hyperparameters: `energy_weight` (not on the same absolute scale as `lambda` — Q2 found SMT's actual solve differs from the textbook objective built from its *own* `full_hess`/`full_jac_dict` by `~3.5-4.8%`, unresolved), `num_ctrl_pts`, `order`, `approx_order`, `smoothness` (per-dimension weights) |
| Multi-D structure | 1-D only, natively; the true multi-D generalization of its *math* is `RBFInterpolator` (isotropic radial energy) | natively N-D, but via a **tensor-product basis and additive, per-dimension energy** $H=\sum_l s_l H_l$ — a third, different structure from both `RBFInterpolator`'s isotropic route and `csaps`'s per-axis-smoothing tensor route |
| Exact GP/variance correspondence | **yes** — Corollary 5 (mean) and this document's own `saddle_point_variance` (variance), both numerically confirmed | **yes, via a different theorem** — Proposition 6, exact for the *idealized* finite-basis objective, but with the solver gap above meaning the *actual* RMTB/RMTC output is not bit-exact to that idealized objective |

**The throughline, not a new finding but worth stating plainly here:** these
differences are not incidental implementation details — they are the reason Q4's
attempt to connect an isotropic Kriging limit to RMTB's 2-D surface failed to
converge (`test_hypotheses_2D_RANS.ipynb`, and this document's own Part 3 of
`spline_variance_2d_vs_smt.ipynb`). RMTS's additive, tensor-product energy is a
**different mathematical object** from both the isotropic thin-plate energy
Corollary 5's natural multi-D extension would need, and from `make_smoothing_spline`'s
own exact 1-D construction — not merely a numerically harder version of the same
thing.

---

### 10) What is `iwp2_kernel` + `saddle_point_solve`, actually? Is it a GP or a spline fit?

**Short answer: it is provably, exactly both at once — that identity is not a
coincidence, it is the entire mathematical content this project is built on.**

**The methodology, precisely.** `iwp2_kernel(x,x') = min(x,x')^2(3\max(x,x')-
\min(x,x'))/6` is the reproducing kernel of the "rough part" of the function space
for the $m=2$ penalty $\int(f'')^2dx$ (Kimeldorf & Wahba, 1970 — the covariance of
*twice-integrated Brownian motion*). `saddle_point_solve(K, P, y, nugget)` solves
Theorem 2a's bordered system

$$\begin{bmatrix}K+\text{nugget}\cdot I & P\\ P^\top & 0\end{bmatrix}
\begin{bmatrix}a\\ \beta\end{bmatrix} = \begin{bmatrix}y\\ 0\end{bmatrix}$$

with `K=iwp2_kernel`, `P=`degree-1 trend, `nugget=lambda`. This is the representer-
theorem solution to minimizing $\|y-f(X)\|^2 + \lambda\int(f'')^2dx$ over
$f=P\beta+h$ — i.e. it does not *approximate* the natural cubic smoothing spline,
it *is* its normal-equations solve, via a different (kernel/bordered-system) route
than `scipy`'s specialized banded algorithm.

**Why compared against `make_smoothing_spline` at all:** because that comparison is
a test of a *mathematical identity* (Kimeldorf–Wahba, via Theorem 2a), not a
"how similar are two different things" exercise — confirmed in
`spline_variance_without_kriging.ipynb` (Q5) to `~1e-12`, i.e. the two routes agree
to the limits of floating-point arithmetic, exactly as an identity (rather than an
approximation) predicts.

**GP or spline fit — both, exactly, simultaneously:**
- **As a spline fit:** the representer theorem shows the minimizer of the penalized
  functional above is a natural cubic spline; the bordered system is that spline's
  normal equations in a particular (kernel) basis.
- **As a Gaussian process:** the *same* minimizer is exactly the posterior mean of
  a Bayesian model — $w\sim\mathcal N(0,s^2A^{-1})$ (improper/diffuse prior on the
  degree-1 trend, proper GP prior with covariance `iwp2_kernel` on the rough part),
  $y_i\mid f\sim\mathcal N(f(x_i),\lambda s^2)$ — with `saddle_point_variance`
  (same `K`, Theorem 2b) giving that GP's actual posterior variance, not an analogy
  to one.

These are not two compatible descriptions of similar objects; Kimeldorf & Wahba's
theorem is precisely the statement that penalized-regression-minimizer and
GP-posterior-mean are the *same function*, for this specific kernel/prior pairing.
That equivalence is why Corollary 5, Proposition 6, and this project's entire
"variance without fitting Kriging" line of work (Q5–Q9) are even meaningful
questions to ask.

---

### 11) Why does `krig_splines.kernels.matern32` differ from SMT's `KRG(corr="matern32")`? What's missing?

**Answer: three concrete, already-found-and-fixed conventions (not missing
*capability*, missing *convention-matching*), plus one genuine, large, missing
*capability* — automatic hyperparameter fitting.** All verified directly in
`spline_variance_without_kriging.ipynb` Part 4, not assumed here:

**The three convention differences** (closing them took SMT's own `optimal_theta`,
`X_offset`/`X_scale`, and `optimal_par["sigma2"]` attributes, plus reading
`krg_based.py`'s source directly):
1. **SMT normalizes `X`** internally, `(X-X_offset)/X_scale` (mean/std of the
   training data), before applying `theta` — our own `matern32(d, kappa)` operates
   on raw, unnormalized `d`. To match, pass `theta = kappa_raw * X_scale` to SMT,
   not `kappa_raw` directly.
2. **SMT's `matern32` has an extra $\sqrt3$** baked into its formula — confirmed by
   reading `Matern32.__call__`'s source: $r=(1+\sqrt3\,\theta d)e^{-\sqrt3\,\theta
   d}$, not $(1+\theta d)e^{-\theta d}$ the way `krig_splines.kernels.matern32`
   (and the paper's own Table 1 convention) defines it. `theta_smt =
   \kappa_{raw}\cdot X_{scale}/\sqrt3` is the full, confirmed translation —
   verified by reconstructing SMT's own correlation matrix from its Cholesky
   factor and matching it to `krig_splines`'s `matern32` to `4e-16`.
3. **SMT's `nugget` option is a numerical jitter** (default `~2.2e-14`), **not**
   the $\tau^2/\sigma^2$ ratio `saddle_point_solve`'s `nugget` argument always is.
   The real noise-to-signal ratio needs `eval_noise=True` + `noise0=[...]`, and
   stays unoptimized only if `noise_bounds` is also collapsed to a point.

**What's genuinely missing — not a convention, a capability:** `krig_splines` has
**no hyperparameter optimizer at all**. `saddle_point_solve` is a pure *solver*:
given `kappa`/`nugget`, it solves the linear system, nothing more. SMT's `KRG` is a
full *estimator*: by default it also **fits** `theta` (and, if `eval_noise=True`,
noise) via maximum likelihood, with `hyper_opt` (`TNC`/`Cobyla`), `theta_bounds`,
and `n_start` (multistart) controlling that search. Every comparison in this
document between "generic auto-tuned `KRG`" and a forced-`kappa` construction is
precisely a comparison between SMT doing this extra work and this repo doing none
of it — not a numerical artifact, the intended difference.

**Smaller, real capability gaps, for completeness:**
- **Heteroscedastic/per-point noise**: `saddle_point_solve` supports this via its
  own `weights` argument (`Winv=diag(1/weights)`) — not missing, just not exposed
  the way SMT's `noise0`-as-a-vector/`use_het_noise` is.
- **Anisotropic/multi-D `theta`**: SMT's `theta` is a per-dimension vector in
  N-D (confirmed in `spline_variance_2d_vs_smt.ipynb`, where `optimal_theta` came
  back as a 2-vector); `krig_splines.kernels.matern32` takes a single scalar
  `kappa` applied to whatever distance you hand it — isotropic unless the caller
  builds an anisotropic kernel by hand (as Part 3/7 of that notebook did).
- **Mixed categorical/hierarchical design spaces, multi-output, derivatives** — SMT
  features with no `krig_splines` counterpart at all, irrelevant to the pure
  continuous-input, single-output regression this project uses but real
  differences in scope, not bugs.
- **Jitter-based conditioning** (SMT's tiny default `nugget`) vs. **diagonal
  equilibration** (`saddle_point_solve`'s own fix, Phase A1) — two different
  strategies for the same near-singularity problem, neither "missing" from the
  other, just different design choices already made deliberately in this repo.

---

### 12) Can a fitted `RBFInterpolator` be converted directly to a B-spline or NURBS representation?

**Answer: no, not exactly, in general — and the reason is a hard mathematical
one, not a missing scipy feature.** An *approximate* conversion is possible and
was verified here; an *exact* one is only possible for specific kernel choices,
consistent with Q8's finding that `RBFInterpolator` and B-splines are genuinely
different representations.

**Why an exact conversion generally cannot exist.** B-splines and NURBS are
**piecewise polynomial** (NURBS: piecewise *rational* polynomial) functions. A
fitted `RBFInterpolator` evaluates as $f(x)=\sum_i a_i K(x,y_i) + P(x)^\top b$
(Q8). For `kernel="thin_plate_spline"` ($K(r)=r^2\log r$), each term is
**transcendental** — $r^2\log r$ is not a polynomial, rational function, or
piecewise version of either, on any interval containing $r=0$. A sum of
transcendental terms is, in general, not exactly representable in *any*
finite-dimensional B-spline/NURBS space, no matter how many knots are allowed —
this is a statement about which function space the fitted model lives in, not a
limitation of any particular software.

**The one real exception, and it connects directly to this project's own 1-D
work:** scipy's `RBFInterpolator` also offers `kernel="cubic"` ($K(r)=r^3$) and
`kernel="linear"` ($K(r)=r$) — genuine **piecewise polynomials** in 1-D (cubic and
linear respectively, with simple knots at the training points). The
`one_d_smoothing_method_comparison.ipynb` work already confirmed
`RBFInterpolator(kernel="cubic")` matches `make_smoothing_spline` to `~1.4e-9` —
and `make_smoothing_spline` itself already *returns* a genuine
`scipy.interpolate.BSpline` object (real `t`, `c`, `k`, confirmed in Q8). So for
this one specific 1-D kernel choice, an exact B-spline representation doesn't just
exist in principle, it's the *same object* scipy already computes a different way.

**The practical, approximate route — verified here, not assumed:** for kernels
without an exact polynomial form (`thin_plate_spline`, `gaussian`,
`multiquadric`, ...), the standard technique is to evaluate the fitted
`RBFInterpolator` on a dense grid and re-fit a genuine B-spline through those
values (`SmoothBivariateSpline`/`bisplrep`, Part 5's tool, with `s` small). Tested
just now, 2-D, `kernel="thin_plate_spline"`: re-fitting through a $25\times25$
grid of RBF-predicted values gives `max|RBF − refit B-spline| ≈ 2.24e-3` against a
data range of `1.80` (`~0.12%`) — a genuine, real B-spline (confirmed `21x21`
knots via `get_knots()`), with an approximation error that shrinks as the
sampling grid gets denser, but never reaches exactly zero for this kernel. This is
a **re-fit**, not a conversion — it has its own resolution/knot-placement choices
and its own (small, controllable) error, unlike the `kernel="cubic"` case above
where no such error exists even in principle.

**NURBS specifically:** not tested in this session (no NURBS-fitting package was
exercised here) — flagged as general knowledge, not a verified claim: CAD/CAM
software routinely fits NURBS surfaces to scattered or gridded data via
least-squares, the same approximate-re-fit idea as the `SmoothBivariateSpline`
route above, generalized to rational (not just polynomial) piecewise functions.
Nothing in this project's own stack (`scipy`, `SMT`, `csaps`) was found to expose
a NURBS fitting routine; a dedicated package would be needed and was not
identified or verified here.

---

### One-line summary

| Target | Generic Kriging? | Exact-kernel Kriging? | Status |
|---|---|---|---|
| scipy smoothing spline (1-D) | **Yes**, in the `κ→0` limit (Corollary 5) | — | Confirmed numerically, synthetic + 2 real datasets |
| SMT RMTB / RMTC (1-D) | No (finite-rank vs. full-rank mismatch) | **Yes, exactly** (Proposition 6), *but* SMT's solved output itself deviates a few % from its own textbook objective — gap not yet root-caused | Identity confirmed; solver-vs-theory gap open |
| A different RMTS-like formulation | No (same reason) | **Yes, exactly**, by the same formulation-agnostic proof — confirmed on two structurally different bases (RMTB, RMTC) | Strong evidence; a fully from-scratch non-SMT confirmation (Phase B1) still pending |
| SMT RMTB / RMTC (2-D surface) | Tensor-product κ-limit (Corollary 5 extension): **not demonstrated**, converges to only ~50–80% of CD's range at best | **Yes, essentially exactly** at/near training data (`~2e-9`/`~1e-8`, tighter than the 1-D case), degrading to `~2e-2` in sparse regions | Proposition 6 route confirmed and dimension-agnostic; Corollary 5's 2-D extension is an open problem |
| Spline uncertainty, without fitting Kriging | — | **Yes, in closed form** — a Green's-function GP (Kimeldorf–Wahba), no free kernel hyperparameter to fit | Implemented (`iwp2_kernel`, `saddle_point_variance`) and numerically confirmed (mean `~1e-12`; variance within `0.02%` after discovering a needed `4κ³` kernel rescaling) |
| 2-D spline uncertainty (`scipy.RBFInterpolator`, thin-plate) | — | **Yes** — same Theorem 2a/2b machinery, dimension-agnostic; scipy's own `RBFInterpolator` docstring *is* Theorem 2a's bordered system | Verified ad hoc (mean `~6e-15` vs. `RBFInterpolator`; variance well-behaved); not yet a `krig_splines` kernel. **Answers a different question than Q4** — not RMTB-specific |
| Extending `make_smoothing_spline`/`UnivariateSpline` with variance | — | **Yes** — wrap, don't patch: scipy for the mean, `saddle_point_variance` for the variance | Recipe sketched and spot-checked; `UnivariateSpline` needs its effective `λ` recovered first (not exposed by scipy) — untested. Wrapper class not yet committed |
| `RBFInterpolator` (thin-plate) is a B-spline? | — | **No** — a global radial-kernel + polynomial predictor, not a knot/control-point tensor-product B-spline | Verified by inspecting both objects' real attributes; no knots exist on `RBFInterpolator`, `SmoothBivariateSpline` has them instead |
| Direct N-D analogue of `make_smoothing_spline` | — | **Yes, two of them** — `RBFInterpolator` (isotropic, any N, exact Theorem 2a match) and the third-party `csaps` package (tensor-product, grid-only, per-axis smoothing — structurally closer to RMTS) | `csaps` installed and run directly for this answer; RMTS differs from `make_smoothing_spline` in basis, energy exactness, solver (iterative, confirmed non-convergent in extreme regimes), and smoothness-control structure |
| `iwp2_kernel`+`saddle_point_solve`: GP or spline? | — | **Both, exactly, simultaneously** — Kimeldorf–Wahba's theorem, not an analogy | Confirmed to `~1e-12` against `make_smoothing_spline` (Q5); `saddle_point_variance` gives the same object's real GP posterior variance |
| Why `krig_splines.matern32` ≠ SMT `KRG(matern32)` | — | **Three fixable conventions** (X-normalization, a hidden `√3`, nugget-vs-jitter) **+ one real capability gap**: no hyperparameter optimizer in `krig_splines` | All three conventions confirmed via SMT's own attributes/source; kernel matrices matched to `4e-16` once translated |
| `RBFInterpolator` → B-spline/NURBS, directly? | — | **No, in general** (transcendental kernel, not piecewise polynomial) — **except `kernel="cubic"`/`"linear"` in 1-D**, which are exact B-splines already | Approximate re-fit verified (`~0.12%` error, grid+`SmoothBivariateSpline`); NURBS conversion not tested, general knowledge only |
