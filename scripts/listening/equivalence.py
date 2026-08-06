"""Dependency-free equivalence / tail statistics for the franken_tts listening protocol.

Implements exactly what `docs/CONFORMANCE_AND_LISTENING.md` binds a gate to:

* TOST equivalence and one-sided non-inferiority tests on clustered listening data;
* the three-way verdict (PASS_EQUIVALENT / FAIL_DIFFERENT / INSUFFICIENT_POWER) so that
  failure-to-reject is never reported as equivalence (plan section 9.4);
* ICC / design-effect diagnostics for the hierarchical structure
  (listener x speaker x text x language);
* CVaR tail statistics for the AF-2 release bound (plan section 10.7).

Standard library only, on purpose: this harness must run in CI, on a listening-lab
laptop, and inside the release gate without a Python environment to install.
Bead: frankentts-v-listening-25m.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Literal, Sequence

__all__ = [
    "student_t_cdf",
    "student_t_ppf",
    "norm_ppf",
    "TostResult",
    "tost",
    "non_inferiority",
    "IccResult",
    "icc_oneway",
    "cvar",
    "TailResult",
    "tail_gate",
    "required_n_tost",
    "achieved_power_tost",
    "group_by",
]

# --------------------------------------------------------------------------------------
# Distribution primitives
# --------------------------------------------------------------------------------------

_BETACF_MAX_ITER = 400
_BETACF_EPS = 3.0e-16
_TINY = 1.0e-300


def _betacf(a: float, b: float, x: float) -> float:
    """Continued-fraction expansion for the incomplete beta function (Lentz's method)."""
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < _TINY:
        d = _TINY
    d = 1.0 / d
    h = d
    for m in range(1, _BETACF_MAX_ITER + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < _TINY:
            d = _TINY
        c = 1.0 + aa / c
        if abs(c) < _TINY:
            c = _TINY
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < _TINY:
            d = _TINY
        c = 1.0 + aa / c
        if abs(c) < _TINY:
            c = _TINY
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < _BETACF_EPS:
            return h
    raise ArithmeticError(f"incomplete beta did not converge for a={a}, b={b}, x={x}")


def _betainc(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta function I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    log_front = (
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log1p(-x)
    )
    front = math.exp(log_front)
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def student_t_cdf(t: float, df: float) -> float:
    """P(T <= t) for Student's t with `df` degrees of freedom."""
    if df <= 0.0:
        raise ValueError("degrees of freedom must be positive")
    if t != t:  # NaN
        raise ValueError("t statistic is NaN")
    if math.isinf(t):
        return 0.0 if t < 0 else 1.0
    x = df / (df + t * t)
    tail = 0.5 * _betainc(0.5 * df, 0.5, x)
    return tail if t <= 0.0 else 1.0 - tail


def student_t_ppf(p: float, df: float) -> float:
    """Inverse CDF of Student's t by bracketed bisection (monotone, so this is exact enough)."""
    if not 0.0 < p < 1.0:
        raise ValueError("p must lie strictly inside (0, 1)")
    lo, hi = -1.0, 1.0
    while student_t_cdf(lo, df) > p:
        lo *= 2.0
        if lo < -1e12:
            return lo
    while student_t_cdf(hi, df) < p:
        hi *= 2.0
        if hi > 1e12:
            return hi
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if student_t_cdf(mid, df) < p:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-13 * max(1.0, abs(mid)):
            break
    return 0.5 * (lo + hi)


def _erfinv(y: float) -> float:
    """Inverse error function via Newton refinement of a rational seed."""
    if not -1.0 < y < 1.0:
        raise ValueError("erfinv argument must lie strictly inside (-1, 1)")
    if y == 0.0:
        return 0.0
    # Winitzki's approximation as the seed, then Newton on erf.
    a = 0.147
    ln1my2 = math.log1p(-y * y)
    term = 2.0 / (math.pi * a) + ln1my2 / 2.0
    x = math.copysign(math.sqrt(math.sqrt(term * term - ln1my2 / a) - term), y)
    for _ in range(4):
        err = math.erf(x) - y
        deriv = 2.0 / math.sqrt(math.pi) * math.exp(-x * x)
        if deriv == 0.0:
            break
        x -= err / deriv
    return x


def norm_ppf(p: float) -> float:
    """Inverse CDF of the standard normal."""
    if not 0.0 < p < 1.0:
        raise ValueError("p must lie strictly inside (0, 1)")
    return math.sqrt(2.0) * _erfinv(2.0 * p - 1.0)


# --------------------------------------------------------------------------------------
# Equivalence testing
# --------------------------------------------------------------------------------------

Decision = Literal["PASS_EQUIVALENT", "FAIL_DIFFERENT", "INSUFFICIENT_POWER"]


@dataclass(frozen=True)
class TostResult:
    """Outcome of a two-one-sided-tests equivalence assessment.

    `decision` is deliberately three-valued. A TOST that fails to reject means "not shown
    equivalent"; whether that is a detected difference or an underpowered panel is decided
    by where the (1 - 2*alpha) confidence interval sits relative to the equivalence band.
    Collapsing INSUFFICIENT_POWER into either PASS or FAIL is the counterfeit-green failure
    mode this protocol exists to prevent.
    """

    n: int
    mean: float
    sd: float
    stderr: float
    df: float
    lower_bound: float
    upper_bound: float
    alpha: float
    ci_low: float
    ci_high: float
    p_lower: float
    p_upper: float
    p_tost: float
    decision: Decision
    required_n: int
    achieved_power: float

    @property
    def passed(self) -> bool:
        return self.decision == "PASS_EQUIVALENT"

    def to_dict(self) -> dict:
        return {
            "n": self.n,
            "mean": self.mean,
            "sd": self.sd,
            "stderr": self.stderr,
            "df": self.df,
            "equivalence_band": [self.lower_bound, self.upper_bound],
            "alpha": self.alpha,
            "ci_1_minus_2alpha": [self.ci_low, self.ci_high],
            "p_lower": self.p_lower,
            "p_upper": self.p_upper,
            "p_tost": self.p_tost,
            "decision": self.decision,
            "required_n_for_declared_power": self.required_n,
            "achieved_power": self.achieved_power,
        }


def _mean_sd(values: Sequence[float]) -> tuple[float, float]:
    n = len(values)
    if n < 2:
        raise ValueError("at least two observations are required")
    mean = math.fsum(values) / n
    ss = math.fsum((v - mean) ** 2 for v in values)
    return mean, math.sqrt(ss / (n - 1))


def required_n_tost(
    sd: float,
    margin: float,
    *,
    alpha: float = 0.05,
    power: float = 0.80,
    true_effect: float = 0.0,
) -> int:
    """Sample size for a TOST at `margin` with the given power, assuming `true_effect`.

    Uses the standard normal approximation n = (z_{1-alpha} + z_{1-beta/2})^2 * sd^2 / (margin - |theta|)^2.
    Returned as a minimum listener/cluster count; the design adds a screening allowance on top.
    """
    slack = margin - abs(true_effect)
    if slack <= 0.0:
        return 2**31 - 1
    if sd <= 0.0:
        return 2
    z_alpha = norm_ppf(1.0 - alpha)
    z_beta = norm_ppf(1.0 - (1.0 - power) / 2.0)
    n = ((z_alpha + z_beta) ** 2) * (sd * sd) / (slack * slack)
    return max(2, math.ceil(n))


def achieved_power_tost(
    n: int,
    sd: float,
    margin: float,
    *,
    alpha: float = 0.05,
    true_effect: float = 0.0,
) -> float:
    """Approximate power of the TOST at the observed n and sd (normal approximation)."""
    slack = margin - abs(true_effect)
    if slack <= 0.0 or sd <= 0.0 or n < 2:
        return 0.0
    z_alpha = norm_ppf(1.0 - alpha)
    lam = slack * math.sqrt(n) / sd
    # Power of the intersection of the two one-sided tests, symmetric-effect approximation.
    z = lam - z_alpha
    power = 2.0 * (0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))) - 1.0
    return max(0.0, min(1.0, power))


def tost(
    values: Sequence[float],
    *,
    center: float = 0.0,
    margin: float,
    alpha: float = 0.05,
    power: float = 0.80,
) -> TostResult:
    """Two one-sided tests for equivalence of `mean(values)` to `center` within +/- `margin`.

    `values` are cluster-level summaries (one number per listener, or one per speaker), not
    raw trials: the by-cluster analysis is how this protocol respects the hierarchical
    design without a mixed-model dependency.
    """
    if margin <= 0.0:
        raise ValueError("equivalence margin must be positive")
    n = len(values)
    mean, sd = _mean_sd(values)
    stderr = sd / math.sqrt(n)
    df = float(n - 1)
    low = center - margin
    high = center + margin

    if stderr == 0.0:
        p_lower = 0.0 if mean > low else 1.0
        p_upper = 0.0 if mean < high else 1.0
        ci_low = ci_high = mean
    else:
        t_lower = (mean - low) / stderr
        t_upper = (mean - high) / stderr
        p_lower = 1.0 - student_t_cdf(t_lower, df)  # H0: mean <= low
        p_upper = student_t_cdf(t_upper, df)  # H0: mean >= high
        crit = student_t_ppf(1.0 - alpha, df)
        ci_low = mean - crit * stderr
        ci_high = mean + crit * stderr

    p_tost = max(p_lower, p_upper)
    if p_tost < alpha:
        decision: Decision = "PASS_EQUIVALENT"
    elif ci_low >= high or ci_high <= low:
        decision = "FAIL_DIFFERENT"
    else:
        decision = "INSUFFICIENT_POWER"

    return TostResult(
        n=n,
        mean=mean,
        sd=sd,
        stderr=stderr,
        df=df,
        lower_bound=low,
        upper_bound=high,
        alpha=alpha,
        ci_low=ci_low,
        ci_high=ci_high,
        p_lower=p_lower,
        p_upper=p_upper,
        p_tost=p_tost,
        decision=decision,
        required_n=required_n_tost(sd, margin, alpha=alpha, power=power),
        achieved_power=achieved_power_tost(n, sd, margin, alpha=alpha),
    )


def non_inferiority(
    values: Sequence[float],
    *,
    center: float = 0.0,
    margin: float,
    worse_is: Literal["higher", "lower"],
    alpha: float = 0.05,
    power: float = 0.80,
) -> TostResult:
    """One-sided non-inferiority test.

    For metrics where only one direction is a regression (WER may not rise by more than
    `margin`; identity rate may not fall by more than `margin`). The unused bound is set to
    infinity so the same `TostResult` shape carries both test kinds into the verdict file.
    """
    n = len(values)
    mean, sd = _mean_sd(values)
    stderr = sd / math.sqrt(n)
    df = float(n - 1)
    crit = student_t_ppf(1.0 - alpha, df)
    ci_low = mean - crit * stderr if stderr else mean
    ci_high = mean + crit * stderr if stderr else mean

    if worse_is == "higher":
        low, high = -math.inf, center + margin
        p_upper = student_t_cdf((mean - high) / stderr, df) if stderr else (0.0 if mean < high else 1.0)
        p_lower = 0.0
        failed_outright = ci_low >= high
    else:
        low, high = center - margin, math.inf
        p_lower = 1.0 - student_t_cdf((mean - low) / stderr, df) if stderr else (0.0 if mean > low else 1.0)
        p_upper = 0.0
        failed_outright = ci_high <= low

    p_tost = max(p_lower, p_upper)
    if p_tost < alpha:
        decision: Decision = "PASS_EQUIVALENT"
    elif failed_outright:
        decision = "FAIL_DIFFERENT"
    else:
        decision = "INSUFFICIENT_POWER"

    return TostResult(
        n=n,
        mean=mean,
        sd=sd,
        stderr=stderr,
        df=df,
        lower_bound=low,
        upper_bound=high,
        alpha=alpha,
        ci_low=ci_low,
        ci_high=ci_high,
        p_lower=p_lower,
        p_upper=p_upper,
        p_tost=p_tost,
        decision=decision,
        required_n=required_n_tost(sd, margin, alpha=alpha, power=power),
        achieved_power=achieved_power_tost(n, sd, margin, alpha=alpha),
    )


# --------------------------------------------------------------------------------------
# Hierarchical-structure diagnostics
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class IccResult:
    """One-way random-effects ICC(1) plus the design effect it implies."""

    k_clusters: int
    n_total: int
    mean_cluster_size: float
    icc: float
    design_effect: float
    n_effective: float

    def to_dict(self) -> dict:
        return {
            "k_clusters": self.k_clusters,
            "n_total": self.n_total,
            "mean_cluster_size": self.mean_cluster_size,
            "icc": self.icc,
            "design_effect": self.design_effect,
            "n_effective": self.n_effective,
        }


def icc_oneway(groups: Sequence[Sequence[float]]) -> IccResult:
    """ICC(1) over clustered observations; `groups[i]` is one cluster (e.g. one listener)."""
    clusters = [list(g) for g in groups if len(g) > 0]
    k = len(clusters)
    n_total = sum(len(g) for g in clusters)
    if k < 2 or n_total <= k:
        return IccResult(k, n_total, n_total / max(k, 1), 0.0, 1.0, float(n_total))

    grand = math.fsum(math.fsum(g) for g in clusters) / n_total
    ms_between = math.fsum(len(g) * (math.fsum(g) / len(g) - grand) ** 2 for g in clusters) / (k - 1)
    ss_within = math.fsum(
        math.fsum((v - math.fsum(g) / len(g)) ** 2 for v in g) for g in clusters
    )
    ms_within = ss_within / (n_total - k)

    sum_sq = math.fsum(len(g) ** 2 for g in clusters)
    n0 = (n_total - sum_sq / n_total) / (k - 1)
    denom = ms_between + (n0 - 1.0) * ms_within
    icc = 0.0 if denom == 0.0 else (ms_between - ms_within) / denom
    icc = max(0.0, min(1.0, icc))

    mean_size = n_total / k
    deff = 1.0 + (mean_size - 1.0) * icc
    return IccResult(k, n_total, mean_size, icc, deff, n_total / deff)


# --------------------------------------------------------------------------------------
# Tail risk (AF-2)
# --------------------------------------------------------------------------------------


def cvar(values: Sequence[float], alpha: float, *, tail: Literal["upper", "lower"]) -> float:
    """Conditional value at risk: the mean of the worst `alpha` fraction of `values`.

    `tail="upper"` for bad-is-high metrics (WER, drift); `tail="lower"` for bad-is-low
    metrics (identity rate, naturalness). At least one item always enters the average, so a
    small corpus degrades to the single worst item rather than silently returning the mean.
    """
    if not values:
        raise ValueError("cvar requires at least one value")
    if not 0.0 < alpha <= 1.0:
        raise ValueError("alpha must lie in (0, 1]")
    ordered = sorted(values, reverse=(tail == "upper"))
    take = max(1, math.ceil(alpha * len(ordered)))
    worst = ordered[:take]
    return math.fsum(worst) / len(worst)


TailDecision = Literal["PASS", "FAIL", "INSUFFICIENT_DATA"]


@dataclass
class TailResult:
    """AF-2 tail gate for one scope (overall, or one canary axis).

    The statistic is CVaR over per-unit values; the threshold is calibrated by a sign-flip
    permutation null rather than fixed in absolute units. An absolute floor is not usable
    here: at realistic cell sizes the per-unit estimate is noise-dominated, so a fixed floor
    either fires on noise or gets widened until it means nothing. Calibrating against the null
    measures EXCESS tail risk, which is what AF-2 is for.
    """

    scope: str
    n_units: int
    n_dropped_units: int
    n_observations: int
    observed_cvar: float
    null_cvar_median: float
    null_quantile_value: float
    threshold: float
    p_value: float
    decision: TailDecision
    notes: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.decision == "PASS"

    def to_dict(self) -> dict:
        return {
            "scope": self.scope,
            "n_units": self.n_units,
            "n_dropped_units": self.n_dropped_units,
            "n_observations": self.n_observations,
            "observed_cvar": self.observed_cvar,
            "null_cvar_median": self.null_cvar_median,
            "null_quantile_value": self.null_quantile_value,
            "threshold": self.threshold,
            "p_value": self.p_value,
            "decision": self.decision,
            "notes": self.notes,
        }


def _quantile(sorted_values: Sequence[float], q: float) -> float:
    """Linear-interpolated quantile of an already-sorted sequence."""
    if not sorted_values:
        raise ValueError("quantile of an empty sequence")
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = q * (len(sorted_values) - 1)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return sorted_values[int(pos)]
    frac = pos - lo
    return sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac


def _unit_cvar(
    observations: Sequence[tuple[str, float]],
    *,
    alpha: float,
    tail: Literal["upper", "lower"],
) -> float:
    buckets: dict[str, list[float]] = {}
    for unit, value in observations:
        buckets.setdefault(unit, []).append(value)
    means = [math.fsum(vals) / len(vals) for vals in buckets.values()]
    return cvar(means, alpha, tail=tail)


def tail_gate(
    observations: Sequence[tuple[str, float]],
    *,
    scope: str,
    center: float,
    alpha: float,
    tail: Literal["upper", "lower"],
    min_obs_per_unit: int,
    max_dropped_unit_fraction: float,
    null_permutations: int,
    null_quantile: float,
    slack: float,
    min_units: int = 5,
    min_observations: int = 0,
    seed: int = 0,
) -> TailResult:
    """Run the calibrated CVaR tail gate over `(unit_key, value)` observations.

    The null flips each paired observation about `center` with probability 1/2, which is the
    exact exchangeability null for paired data: under "the two systems are interchangeable",
    the sign of every paired contrast is a fair coin. Comparing the observed tail against that
    distribution asks the only question worth asking — is the bad tail worse than chance?
    """
    import random as _random

    notes: list[str] = []
    buckets: dict[str, list[float]] = {}
    for unit, value in observations:
        buckets.setdefault(unit, []).append(value)

    kept = {u: v for u, v in buckets.items() if len(v) >= min_obs_per_unit}
    n_dropped = len(buckets) - len(kept)
    total_units = len(buckets)
    if total_units and n_dropped / total_units > max_dropped_unit_fraction:
        notes.append(
            f"UNDER_SAMPLED_UNITS: dropped {n_dropped}/{total_units} units below "
            f"{min_obs_per_unit} observations"
        )
    kept_obs = [(u, v) for u, vals in kept.items() for v in vals]

    if len(kept) < min_units or len(kept_obs) < max(min_observations, 1):
        notes.append(
            f"INSUFFICIENT_UNITS: {len(kept)} usable units, {len(kept_obs)} observations"
        )
        return TailResult(
            scope=scope,
            n_units=len(kept),
            n_dropped_units=n_dropped,
            n_observations=len(kept_obs),
            observed_cvar=float("nan"),
            null_cvar_median=float("nan"),
            null_quantile_value=float("nan"),
            threshold=float("nan"),
            p_value=float("nan"),
            decision="INSUFFICIENT_DATA",
            notes=notes,
        )

    observed = _unit_cvar(kept_obs, alpha=alpha, tail=tail)

    rng = _random.Random(seed)
    null_values: list[float] = []
    for _ in range(null_permutations):
        flipped = [
            (u, (2.0 * center - v) if rng.random() < 0.5 else v) for u, v in kept_obs
        ]
        null_values.append(_unit_cvar(flipped, alpha=alpha, tail=tail))
    null_values.sort()

    q_value = _quantile(null_values, null_quantile)
    if tail == "lower":
        threshold = q_value - slack
        passed = observed >= threshold
        p_value = (1 + sum(1 for v in null_values if v <= observed)) / (len(null_values) + 1)
    else:
        threshold = q_value + slack
        passed = observed <= threshold
        p_value = (1 + sum(1 for v in null_values if v >= observed)) / (len(null_values) + 1)

    if not passed:
        notes.append(
            f"TAIL_BREACH: CVaR {observed:.4f} beyond calibrated threshold {threshold:.4f} "
            f"(null q{null_quantile:g} = {q_value:.4f}, slack {slack:g})"
        )

    if any(note.startswith("UNDER_SAMPLED_UNITS") for note in notes):
        decision: TailDecision = "INSUFFICIENT_DATA" if passed else "FAIL"
    else:
        decision = "PASS" if passed else "FAIL"

    return TailResult(
        scope=scope,
        n_units=len(kept),
        n_dropped_units=n_dropped,
        n_observations=len(kept_obs),
        observed_cvar=observed,
        null_cvar_median=_quantile(null_values, 0.5),
        null_quantile_value=q_value,
        threshold=threshold,
        p_value=p_value,
        decision=decision,
        notes=notes,
    )


def group_by(items: Iterable[dict], key: str) -> dict[str, list[dict]]:
    """Small helper used by the analyzer to build by-listener / by-speaker cluster lists."""
    out: dict[str, list[dict]] = {}
    for item in items:
        out.setdefault(str(item[key]), []).append(item)
    return out
