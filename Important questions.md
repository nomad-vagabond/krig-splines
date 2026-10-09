# Important questions:

1) Can any Kriging model for 1D case produce the same curve (kriging mean) as Scipy smoothing splines? How to proove/disprove this numerically? Is there a mathematical confirmation?
2) Can any Kriging model for 1D case produce the same curve (kriging mean) as SMT's RMTB or RMTC? How to proove/disprove this numerically? Is there a mathematical confirmation?
3) Can any Kriging model for 1D case produce the same curve (kriging mean) as RMTS that has different mathematical formulation than those in SMT? How to proove/disprove this numerically? Is there a mathematical confirmation?
4) Can a Kriging model for 2D case produce the same surface (kriging mean) as SMT's RMTB or RMTC? How to proove/disprove this numerically? Is there a mathematical confirmation?
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

---

### One-line summary

| Target | Generic Kriging? | Exact-kernel Kriging? | Status |
|---|---|---|---|
| scipy smoothing spline (1-D) | **Yes**, in the `κ→0` limit (Corollary 5) | — | Confirmed numerically, synthetic + 2 real datasets |
| SMT RMTB / RMTC (1-D) | No (finite-rank vs. full-rank mismatch) | **Yes, exactly** (Proposition 6), *but* SMT's solved output itself deviates a few % from its own textbook objective — gap not yet root-caused | Identity confirmed; solver-vs-theory gap open |
| A different RMTS-like formulation | No (same reason) | **Yes, exactly**, by the same formulation-agnostic proof — confirmed on two structurally different bases (RMTB, RMTC) | Strong evidence; a fully from-scratch non-SMT confirmation (Phase B1) still pending |
| SMT RMTB / RMTC (2-D surface) | Tensor-product κ-limit (Corollary 5 extension): **not demonstrated**, converges to only ~50–80% of CD's range at best | **Yes, essentially exactly** at/near training data (`~2e-9`/`~1e-8`, tighter than the 1-D case), degrading to `~2e-2` in sparse regions | Proposition 6 route confirmed and dimension-agnostic; Corollary 5's 2-D extension is an open problem |
