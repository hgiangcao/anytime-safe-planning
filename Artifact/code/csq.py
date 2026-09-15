"""Time-uniform (anytime-valid) quantile confidence sequences and baseline calibrators.

Core object: a one-sided confidence sequence (CS) for the beta-quantile q_beta of an
iid/exchangeable score stream s_1, s_2, ...  The CS upper bound U_t satisfies

    P( exists t >= 1 : U_t < q_beta ) <= delta,

i.e. it is valid *uniformly over time*, hence at arbitrary data-dependent stopping /
replanning times (Ville's inequality).  Construction (Howard--Ramdas style mixture
supermartingale, exact, no approximations):

  Let X_i = 1{s_i < q_beta}.  Under continuous F, X_i ~ Bernoulli(beta) iid.
  For the one-sided null H0: p <= beta we use the beta-binomial mixture e-process
  with uniform mixing over the alternative lambda in (beta, 1):

      M_t(Z) = (1/(1-beta)) * Integral_beta^1 (lam/beta)^Z ((1-lam)/(1-beta))^(t-Z) dlam,

  where Z = sum X_i (count of scores strictly below q_beta).  E[M increment] = 1 at
  p = beta and <= 1 for p < beta (monotone likelihood ratio), so M is a nonnegative
  supermartingale with M_0 = 1 under H0; Ville gives P(exists t: M_t >= 1/delta) <= delta.
  Closed form via the regularized incomplete beta function:

      M_t(Z) = B(Z+1, t-Z+1) * (1 - I_beta(Z+1, t-Z+1)) / ((1-beta) beta^Z (1-beta)^(t-Z)).

  The CS upper bound is the order statistic U_t = s_(u_t) with
      u_t = min{ u <= t : M_t(u) >= 1/delta }   (+infinity if no such u),
  because {U_t < q_beta} = {Z_t >= u_t} subset {M_t(Z_t) >= 1/delta} by monotonicity of
  M_t in Z.

All other calibrators (split-CP union bound, ACI, Gaussian/Rayleigh chance constraint)
are implemented here as baselines on the same score stream.
"""
from __future__ import annotations

import bisect
import hashlib
import math
import os

import numpy as np
from scipy.special import betainc, betaln
from scipy.stats import binom


# ----------------------------------------------------------------------------- e-value
def log_evalue(Z, t, beta):
    """log M_t(Z): one-sided beta-binomial mixture e-value against H0: p <= beta.

    Z = number of observations strictly below the candidate quantile value,
    t = number of observations, beta = target quantile level in (0,1).
    Vectorized over Z and t (numpy broadcasting).
    """
    Z = np.asarray(Z, dtype=float)
    t = np.asarray(t, dtype=float)
    a = Z + 1.0
    b = t - Z + 1.0
    # mass of Beta(a,b) above beta, computed via the complement identity
    # 1 - I_beta(a,b) = I_{1-beta}(b,a) to avoid catastrophic cancellation
    sf = betainc(b, a, 1.0 - beta)
    with np.errstate(divide="ignore"):
        return (
            betaln(a, b)
            + np.log(sf)
            - Z * np.log(beta)
            - (t - Z) * np.log1p(-beta)
            - np.log1p(-beta)
        )


def cs_boundary(t_max, beta, delta, cache_dir=None, rho_tot=0.0, rho_step=0.0):
    """Precompute u_t for t = 1..t_max: minimal below-count u such that
    log M_t(u) >= log(1/delta) + slack_t.  u_t = t+1 encodes 'no finite bound'.

    slack_t = rho_tot/beta + t*log1p(rho_step/beta) is the *robustness slack* of
    Prop. 4: it makes the boundary valid not only under exchangeability but under
    a bounded departure from it (see RobustQuantileCS).  rho_tot = rho_step = 0
    recovers the exact exchangeable boundary of Theorem 1.

    Vectorized binary search (M_t(u) is nondecreasing in u).  Cached to disk.
    """
    if cache_dir is not None:
        key = f"bnd_{beta:.10g}_{delta:.6g}_{t_max}_{rho_tot:.10g}_{rho_step:.10g}"
        h = hashlib.md5(key.encode()).hexdigest()[:16]
        path = os.path.join(cache_dir, f"boundary_{h}.npy")
        if os.path.exists(path):
            return np.load(path)
    ts = np.arange(1, t_max + 1, dtype=np.int64)
    lo = np.ceil(beta * ts).astype(np.int64)
    hi = ts + 1  # exclusive sentinel: u==t+1 means infinity
    thresh = (np.log(1.0 / delta) + rho_tot / beta
              + ts * np.log1p(rho_step / beta))
    while True:
        active = lo < hi
        if not np.any(active):
            break
        mid = (lo + hi) // 2
        # only evaluate where active and mid <= t
        ok = np.zeros_like(active)
        idx = np.where(active)[0]
        th = thresh[idx] if np.ndim(thresh) else thresh
        ok_idx = log_evalue(mid[idx], ts[idx], beta) >= th
        ok[idx] = ok_idx
        hi = np.where(active & ok, mid, hi)
        lo = np.where(active & ~ok, mid + 1, lo)
    u = lo  # in [ceil(beta t), t+1]
    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)
        np.save(path, u)
    return u


def alpha_spending_rank(n, beta, level):
    """Smallest exact-binomial upper-tail rank at one predeclared query.

    ``n`` is the *total* number of exchangeable scores available at the query
    (fixed warm pool plus earlier deployment scores), while ``level`` is the
    score-independent error allocation for that deployment query.  The return
    value is in ``[ceil(beta*n), n+1]``; ``n+1`` means that no finite order
    statistic reaches the requested level.

    Keeping the sample-size and spending clocks separate is important.  We do
    not spend alpha on the unqueried prefixes inside the fixed warm pool.
    """
    n = int(n)
    beta = float(beta)
    level = float(level)
    if n < 1:
        raise ValueError("n must be positive")
    if not (0.0 < beta < 1.0 and 0.0 < level < 1.0):
        raise ValueError("beta and level must lie in (0,1)")
    lo = int(math.ceil(beta * n))
    hi = n + 1
    log_level = math.log(level)
    while lo < hi:
        mid = (lo + hi) // 2
        if binom.logsf(mid - 1, n, beta) <= log_level:
            hi = mid
        else:
            lo = mid + 1
    return lo


class QuantileCS:
    """Online time-uniform upper confidence sequence for the beta-quantile.

    radius() at 'time' t (= number of scores absorbed so far) returns U_t with the
    guarantee P(exists t: U_t < q_beta) <= delta.  Scores are absorbed via add().
    A warm-start pool (exchangeable with the stream) may be supplied.
    """

    def __init__(self, beta, delta, warm_scores=None, t_max=200000, cache_dir=None):
        self.beta = float(beta)
        self.delta = float(delta)
        self.rho_tot = 0.0
        self.rho_step = 0.0
        self._cache_dir = cache_dir
        initial_capacity = max(1, int(t_max))
        self._u = cs_boundary(initial_capacity, self.beta, self.delta,
                              cache_dir=cache_dir)
        self.sorted = sorted(float(s) for s in (warm_scores if warm_scores is not None else []))

    def add(self, s):
        bisect.insort(self.sorted, float(s))

    @property
    def t(self):
        return len(self.sorted)

    def _ensure_capacity(self, t):
        """Grow the cached boundary geometrically; initial t_max is not a horizon."""
        if t <= len(self._u):
            return
        new_capacity = max(int(t), 2 * len(self._u))
        self._u = cs_boundary(
            new_capacity, self.beta, self.delta, cache_dir=self._cache_dir,
            rho_tot=self.rho_tot, rho_step=self.rho_step)

    def radius(self):
        t = len(self.sorted)
        if t == 0:
            return np.inf
        self._ensure_capacity(t)
        u = int(self._u[t - 1])  # boundary array is 1-indexed by t
        if u > t:
            return np.inf
        return self.sorted[u - 1]


class AlphaSpendingQuantileCS:
    """Standard exact-binomial alpha-spending quantile CS (G1 baseline).

    The API and target event match :class:`QuantileCS`; only the time-uniform
    construction differs.  Monitoring starts after the fixed warm pool.  At
    deployment-score query ``k`` the class uses total sample size ``N_k`` and
    the predeclared level ``delta/[k(k+1)]``.  Thus the exact fixed-time tests
    plus a union bound establish the same indefinite-horizon G1 event without
    charging alpha to warm-pool prefixes that were never queried.  Score values
    affect the returned order statistic, never the level schedule.
    """

    def __init__(self, beta, delta, warm_scores=None, t_max=200000,
                 cache_dir=None):
        self.beta = float(beta)
        self.delta = float(delta)
        self.sorted = sorted(float(s) for s in (
            warm_scores if warm_scores is not None else []))
        self.warm_size = len(self.sorted)
        # Retained only for API compatibility with the other online
        # calibrators.  This baseline evaluates an exact scalar tail at each
        # queried deployment prefix and needs neither a horizon nor a cache.
        self._t_max_hint = int(t_max)
        self._cache_dir = cache_dir
        self.last_query_index = None
        self.last_alpha = None
        self.last_rank = None

    def add(self, s):
        bisect.insort(self.sorted, float(s))

    @property
    def t(self):
        return len(self.sorted)

    def radius(self):
        n = len(self.sorted)
        if n == 0:
            return np.inf
        # One new score is added after every scored deployment query.  Repeated
        # calls before add() reuse the same fixed-time event and spend no extra
        # alpha.  Unscored identity-uncertain windows likewise do not advance
        # the score-stream clock.
        k = n - self.warm_size + 1
        level = self.delta / (k * (k + 1.0))
        u = alpha_spending_rank(n, self.beta, level)
        self.last_query_index = int(k)
        self.last_alpha = float(level)
        self.last_rank = int(u)
        if u > n:
            return np.inf
        return self.sorted[u - 1]



class RobustQuantileCS(QuantileCS):
    """Ours: shift-robust time-uniform quantile CS (Prop. 4).

    Theorem 1 assumes the deployment scores are exchangeable with the pool.  In
    closed loop they are not: the robot's own motion perturbs the crowd, and a
    cross-scene deployment shifts the score law outright.  Let

        p_t = P(s_t < q_beta | F_{t-1})

    be the *realised* conditional exceedance probability at window t (p_t <= beta
    under exchangeability).  For any mixing weight lam in (beta,1) the per-step
    increment of the mixture e-process has conditional expectation

        p_t lam/beta + (1-p_t)(1-lam)/(1-beta)  <=  1 + (p_t - beta)_+ / beta,

    because the expression is affine and increasing in p_t with slope
    lam/beta - (1-lam)/(1-beta) <= 1/beta on lam in (beta,1).  Mixing preserves
    the bound, so

        M_t * prod_{s<=t} (1 + (p_s-beta)_+/beta)^{-1}

    is a nonnegative supermartingale, and Ville's inequality gives, under a
    *total shift budget*  sum_t (p_t - beta)_+ <= rho_tot,

        P( exists t : M_t >= exp(rho_tot/beta)/delta )  <=  delta,

    since prod (1+x_s) <= exp(sum x_s).  Running the boundary of eq. (3) at the
    inflated threshold log(1/delta) + rho_tot/beta therefore restores time-uniform
    validity, and the degradation is *explicit and constant in t*: tolerating a
    total budget rho_tot costs exactly rho_tot/beta nats of evidence, i.e. it is
    equivalent to shrinking delta to delta*exp(-rho_tot/beta).  A per-step budget
    (p_t <= beta + rho_step for all t) is also supported and costs
    t*log(1+rho_step/beta) nats, which grows linearly -- the honest statement that
    a persistent shift eventually exhausts any fixed confidence.

    rho_tot = rho_step = 0 reduces exactly to QuantileCS.
    """

    def __init__(self, beta, delta, warm_scores=None, rho_tot=0.0, rho_step=0.0,
                 t_max=200000, cache_dir=None):
        self.beta = float(beta)
        self.delta = float(delta)
        self.rho_tot = float(rho_tot)
        self.rho_step = float(rho_step)
        self._cache_dir = cache_dir
        initial_capacity = max(1, int(t_max))
        self._u = cs_boundary(initial_capacity, self.beta, self.delta, cache_dir=cache_dir,
                              rho_tot=self.rho_tot, rho_step=self.rho_step)
        self.sorted = sorted(float(s) for s in (warm_scores if warm_scores is not None else []))


class ViolationBudgetCS:
    """Ours: an anytime-valid upper bound on the CUMULATIVE number of violated
    windows (guarantee G2 of Sec. III-C).

    Motivation.  Theorem 1 certifies a per-window risk eps uniformly in time, and
    Props. 1-2 show that an empirical-rank/per-window-allocation certificate
    loses a finite radius past a horizon fixed by the calibration budget.
    Between those two lies the
    quantity a long deployment actually cares about: how many certified regions
    have failed so far.  This object bounds it, at every horizon, with no horizon
    known in advance and under arbitrary data-dependent stopping.

    Construction.  Let r_w be the radius read *before* window w and
    Yhat_w = 1{ s_w > max(r_w, q_{1-eps}) }.  Yhat_w is dominated by a
    Bernoulli(eps) conditionally on the past *unconditionally on the CS event*
    (the max removes the dependence on whether the CS bound happens to hold), so
    the same beta-binomial mixture e-process -- now with beta := eps and Z :=
    the running Yhat count -- is a supermartingale under H0: P(Yhat=1) <= eps.
    With  B_W = u_W(eps, delta') - 1  read off the SAME precomputed boundary
    array, Ville gives

        P( exists W >= 1 : sum_{w<=W} Yhat_w > B_W )  <=  delta'.

    On the Theorem-1 event {for all t: U_t >= q_{1-eps}} (probability >= 1-delta)
    the realised violations coincide with Yhat, so

        P( exists W : V_W > B_W )  <=  delta + delta',      V_W = #violations,

    simultaneously over all W and all stopping times.  B_W/W -> eps from above
    with B_W = eps W + O(sqrt(W log W)) for the displayed uniform mixture, so
    the bound is a genuine failure-count budget rather than the vacuous
    "at least one violation is near-certain".

    The same boundary routine produces both objects, but because the parameters
    are (beta,delta) for the radius and (eps,delta') for the count, they are two
    separately precomputed arrays.
    """

    def __init__(self, eps, delta_prime, w_max=100000, cache_dir=None):
        self.eps = float(eps)
        self.delta_prime = float(delta_prime)
        self._cache_dir = cache_dir
        initial_capacity = max(1, int(w_max))
        self._u = cs_boundary(initial_capacity, self.eps, self.delta_prime,
                              cache_dir=cache_dir)
        self.count = 0
        self.W = 0

    def _ensure_capacity(self, W):
        """Grow the cached count boundary geometrically on an indefinite stream."""
        if W <= len(self._u):
            return
        new_capacity = max(int(W), 2 * len(self._u))
        self._u = cs_boundary(new_capacity, self.eps, self.delta_prime,
                              cache_dir=self._cache_dir)

    def budget(self, W=None):
        """B_W: the largest violation count not rejected at level delta' by
        window W (+inf before any window)."""
        W = self.W if W is None else int(W)
        if W < 1:
            return np.inf
        self._ensure_capacity(W)
        return int(self._u[W - 1]) - 1

    def add(self, violated):
        self.W += 1
        self.count += int(bool(violated))
        return self.count <= self.budget()

    @property
    def exceeded(self):
        return self.count > self.budget()


def violation_budget_curve(eps, delta_prime, w_max, cache_dir=None):
    """B_W for W = 1..w_max (vectorised; used by the figures and by the tests)."""
    return cs_boundary(w_max, eps, delta_prime, cache_dir=cache_dir) - 1


# ------------------------------------------------------------------- baseline calibrators
def split_cp_quantile(sorted_scores, level):
    """Finite-sample split-conformal quantile: the ceil((n+1)*level)-th order statistic.
    Returns +inf when the index exceeds n (the union-bound blow-up regime)."""
    n = len(sorted_scores)
    if n == 0:
        return np.inf
    k = int(np.ceil((n + 1) * level))
    if k > n:
        return np.inf
    return sorted_scores[k - 1]


class UnionBoundCP:
    """Lindemann-et-al.-style union-bound conformal calibrator (reimplementation).

    Fixed calibration set; per-step level 1 - delta/T so that a union bound over the
    T-step episode gives episode miscoverage <= delta.  No online accretion (the
    split-CP guarantee does not survive data-dependent stopping/adaptation)."""

    def __init__(self, delta, T, warm_scores):
        self.sorted = np.sort(np.asarray(warm_scores, dtype=float))
        self.level = 1.0 - delta / T
        self._r = split_cp_quantile(self.sorted, self.level)

    def add(self, s):  # no-op: fixed calibration
        pass

    def radius(self):
        return self._r


class AnytimeUnionCP:
    """Ours (episode-level variant): anytime union conformal.

    Per-episode guarantee P(any scored window violated) <= delta for episodes of
    ARBITRARY, unknown length: window w (w = 1, 2, ... within the episode) uses the
    split-CP quantile at level 1 - eps_w with eps_w = delta / (w (w+1)), so
    sum_w eps_w = delta.  Each window is marginally valid under exchangeability of
    (pool + new score); the union bound needs no independence.  The pool accretes
    online (exchangeable only under the stated iid premise; replay format alone
    does not establish it), so late windows' extreme quantiles become certifiable
    as data accumulates under that premise.  Contrast: the
    fixed-T union bound must know T and returns +inf once delta/T < 1/(n+1)."""

    def __init__(self, delta, warm_scores):
        self.delta = float(delta)
        self.sorted = sorted(float(s) for s in warm_scores)
        self.w = 1

    def new_episode(self):
        self.w = 1

    def add(self, s):
        bisect.insort(self.sorted, float(s))
        self.w += 1

    def radius(self):
        eps_w = self.delta / (self.w * (self.w + 1.0))
        return split_cp_quantile(self.sorted, 1.0 - eps_w)


class ACI:
    """Adaptive conformal inference calibrator (Gibbs & Candes; Dixit et al. planner).

    alpha_{t+1} = alpha_t + gamma * (alpha_target - err_t), err_t = 1{s > r}.
    Radius = split-CP quantile of (warm + online) scores at level 1 - alpha_t.
    Only asymptotic average-coverage guarantees; included as baseline."""

    def __init__(self, alpha_target, warm_scores, gamma=0.005):
        self.alpha_target = float(alpha_target)
        self.alpha = float(alpha_target)
        self.gamma = float(gamma)
        self.sorted = sorted(float(s) for s in warm_scores)

    def radius(self):
        a = self.alpha
        if a <= 0.0:
            return np.inf
        if a >= 1.0:
            return 0.0
        return split_cp_quantile(self.sorted, 1.0 - a)

    def update_err(self, err):
        self.alpha += self.gamma * (self.alpha_target - float(err))

    def add(self, s):
        bisect.insort(self.sorted, float(s))


class RayleighCC:
    """Gaussian chance-constrained baseline via moment matching.

    2-D prediction residual with isotropic Gaussian components => residual norm is
    Rayleigh(sigma).  Fit sigma by moment matching on calibration scores; radius is
    the (1 - delta/T) Rayleigh quantile (per-step chance constraint, union over T)."""

    def __init__(self, delta, T, warm_scores):
        s = np.asarray(warm_scores, dtype=float)
        self.sigma = float(np.sqrt(np.mean(s**2) / 2.0))
        p_miss = delta / T
        self._r = self.sigma * np.sqrt(-2.0 * np.log(p_miss))

    def add(self, s):
        pass

    def radius(self):
        return self._r


class Uncalibrated:
    """Raw-prediction floor: no inflation."""

    def add(self, s):
        pass

    def radius(self):
        return 0.0



def allocation_grid_heuristic(quantile_fn, T, delta, n, grid_size=6000):
    """Lagrangian/grid heuristic for per-window alpha allocation.

    Choose eps_1..eps_T >= 1/(n+1) with sum_w eps_w <= delta minimising the total
    certified radius sum_w Q_w(1 - eps_w), where Q_w is window w's empirical score
    quantile function (quantile_fn(w, level) -> radius).  The routine searches
    Lagrangian water-filling candidates

        eps_w(nu) = argmin_eps [ Q_w(1-eps) + nu * eps ],

    on a log-spaced grid and returns the better of this candidate and uniform
    allocation.  Empirical order-statistic curves are nonconvex step functions,
    so a Lagrangian duality gap can occur and this is *not* a global optimiser.

    Returns (eps vector, radii vector, feasible: bool).  feasible=False iff some
    window cannot be certified at all, which -- by Prop. 1 -- happens exactly when
    T > delta(n+1), *whatever* the allocation.  That feasibility wall is checked
    analytically before the heuristic and does not rely on radius optimisation.
    """
    floor = 1.0 / (n + 1.0)
    if T * floor > delta * (1 + 1e-12):
        return None, None, False
    grid = np.unique(np.concatenate([
        np.geomspace(floor, delta, grid_size), [floor, delta]]))
    grid = grid[(grid >= floor - 1e-15) & (grid <= delta + 1e-15)]
    row0 = np.array([quantile_fn(0, 1.0 - g) for g in grid])
    Q = np.stack([row0] + [np.array([quantile_fn(w, 1.0 - g) for g in grid])
                           for w in range(1, T)]) if T > 1 else row0[None]  # [T,G]
    Q = np.where(np.isfinite(Q), Q, np.inf)

    def eps_at(nu):
        j = np.argmin(Q + nu * grid[None, :], axis=1)
        return grid[j], j

    lo, hi = 0.0, 1.0
    while eps_at(hi)[0].sum() > delta and hi < 1e18:
        hi *= 4.0
    for _ in range(200):                       # bisection on the multiplier
        mid = 0.5 * (lo + hi)
        if eps_at(mid)[0].sum() > delta:
            lo = mid
        else:
            hi = mid
    eps, j = eps_at(hi)
    radii = Q[np.arange(T), j]
    eu = np.full(T, delta / T)
    ru = np.array([quantile_fn(w, 1.0 - delta / T) for w in range(T)])
    if np.all(np.isfinite(ru)) and ru.sum() < radii.sum():
        eps, radii = eu, ru
    return eps, radii, bool(np.all(np.isfinite(radii)))


def cert_horizon_lookahead(n, delta, H):
    """Certifiable horizon when the alpha budget is additionally split across the
    H lookaheads of each window (the axis Cleaveland et al. optimise over).

    Each (window, lookahead) pair then needs its own order statistic, i.e.
    eps_{w,j} >= 1/(n+1), so T*H/(n+1) <= delta and the wall MOVES DOWN by a
    factor H: T <= delta(n+1)/H.  Per-lookahead allocation therefore buys tighter
    tubes at an H-fold increase in the data requirement -- the opposite of what
    a conservatism-reduction narrative suggests at long horizons."""
    return int(np.floor(delta * (n + 1) / float(H)))


# --------------------------------------------------------- certifiable horizons
def cert_horizon_fixed_T(n, delta):
    """Largest episode horizon T certifiable by ANY alpha-allocation over a KNOWN
    T from a pool of n scores.  A window at level 1-eps needs the
    ceil((n+1)(1-eps))-th order statistic to exist, i.e. eps >= 1/(n+1); summing
    the T per-window levels to at most delta forces T/(n+1) <= delta.  Hence the
    fixed-T union bound's threshold T <= delta(n+1) is allocation-independent."""
    return int(np.floor(delta * (n + 1)))


def cert_horizon_schedule(eps_of_w, n0, delta, w_max=100000, accrete=1):
    """Number of leading windows a horizon-free spending schedule can certify.

    eps_of_w(w) -> per-window level for window w = 1, 2, ...  The pool starts at
    n0 and grows by `accrete` scores per window (one scored window each).  Window
    w is certifiable iff eps_w >= 1/(n_w+1) AND the cumulative spend stays <=
    delta.  Returns the largest W with all of 1..W certifiable."""
    spent, w = 0.0, 0
    while w < w_max:
        w += 1
        n_w = n0 + accrete * (w - 1)
        e = float(eps_of_w(w))
        if e < 1.0 / (n_w + 1) or spent + e > delta * (1 + 1e-12):
            return w - 1
        spent += e
    return w_max


def harmonic_spend(n0, W, accrete=1):
    """Exact minimal alpha spend for W leading windows of an accreting pool."""
    if n0 < 0 or W < 0 or accrete < 1:
        raise ValueError("n0 and W must be nonnegative and accrete must be positive")
    return math.fsum(
        1.0 / (n0 + accrete * (w - 1) + 1.0) for w in range(1, W + 1))


def cert_horizon_minimal(n0, delta, accrete=1):
    """Optimal horizon inside the empirical-rank/per-window-allocation family:
    spend the least the pool allows,
    eps_w = 1/(n_w+1), stopping before the first unaffordable window.  For
    accrete=1 the exact integer cutoff is

        max {W >= 0 : H_(n0+W) - H_n0 <= delta}.

    The logarithmic expressions are only integral bounds: n0*expm1(delta) is a
    sufficient real-valued cutoff, while (n0+1)*expm1(delta) is an upper bound.
    Neither replaces the discrete harmonic test (e.g. n0=1, delta=1/2 gives
    W*=1 although n0*expm1(delta)<1)."""
    return cert_horizon_schedule(lambda w: 1.0 / (n0 + accrete * (w - 1) + 1),
                                 n0, delta, accrete=accrete)


class BudgetedAUC:
    """Ours (episode-level, horizon-free, budget-aware): fixes AnytimeUnionCP's
    short certifiable horizon.

    AnytimeUnionCP spends eps_w = delta/(w(w+1)): half the budget goes to the
    first window, and eps_w decays like w^-2 while the pool grows only one score
    per window, so certification dies at W ~ sqrt(delta n) (Prop. 3).  Here the
    spend is *budget-aware*: with remaining budget d_rem (reset to delta at every
    episode start) and current pool size n_w,

        eps_w = max( 1/(n_w+1),  lam * d_rem )      while  d_rem >= 1/(n_w+1),

    and radius() returns +inf once the budget can no longer buy even the pool's
    most extreme order statistic.  sum_w eps_w <= delta holds by construction, so
    Theorem 2's union bound applies verbatim: P(exists w <= W: s_w > r_w) <= delta
    for episodes of arbitrary unknown length W.  lam in (0,1) traverses the
    radius-horizon frontier: lam=0 attains the exact harmonic cutoff
    max{W: sum_{w<=W}1/(n+w)<=delta}; larger lam buys tighter early radii at the
    cost of horizon.  Once the next floor is unaffordable no further budget is
    spent and every later radius remains infinite.

    The implemented levels are fixed before the corresponding score is seen:
    they are score-value-independent functions only of the query index, current
    pool *size*, ``lam``, and the deterministic remaining-budget recursion.
    Marginal exchangeability alone would not justify arbitrary score-adaptive
    level choices; such an extension would require a stronger conditional rank
    premise and is not claimed here."""

    def __init__(self, delta, warm_scores, lam=0.10):
        self.delta = float(delta)
        self.lam = float(lam)
        self.sorted = sorted(float(s) for s in warm_scores)
        self.w = 1
        self.d_rem = self.delta
        self._cached_w = 0
        self._cached_r = np.inf
        self.exhausted = False

    def new_episode(self):
        self.w = 1
        self.d_rem = self.delta
        self._cached_w = 0
        self._cached_r = np.inf
        self.exhausted = False

    def add(self, s):
        bisect.insort(self.sorted, float(s))
        self.w += 1

    def _advance(self):
        if self.exhausted:
            self._cached_r = np.inf
            self._cached_w = self.w
            return
        n_w = len(self.sorted)
        floor = 1.0 / (n_w + 1.0)
        if self.d_rem < floor:
            self._cached_r = np.inf
            self.exhausted = True
        else:
            eps = max(floor, self.lam * self.d_rem)
            self.d_rem -= eps
            self._cached_r = split_cp_quantile(self.sorted, 1.0 - eps)
        self._cached_w = self.w

    def radius(self):
        if self._cached_w != self.w:
            self._advance()
        return self._cached_r


class ACIEpisode(ACI):
    """Episode-targeted ACI baseline: the Gibbs-Candes update run at the
    Bonferroni window level alpha = delta / W so that the *episode* miscoverage
    target is delta.  Included because judging a window-targeted ACI against an
    episode-level target is arithmetic (1-(1-delta)^W), not a discovered failure;
    the substantive claim is that ACI has no finite-sample episode guarantee at
    ANY tuning, which this arm makes testable.

    The step size is scaled down to gamma <= alpha_target/5: ACI's level performs
    a random walk with stationary spread ~sqrt(gamma alpha/2), so keeping the
    default gamma at a Bonferroni target would push alpha below zero (no finite
    radius) a large fraction of the time.  Scaling gamma is a fairness measure
    for the baseline, not a handicap."""

    def __init__(self, delta, W, warm_scores, gamma=0.01):
        a = delta / float(W)
        super().__init__(a, warm_scores, gamma=min(gamma, a / 5.0))
