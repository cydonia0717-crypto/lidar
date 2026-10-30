"""Independent event-risk baselines with finite-bank, finite-look confidence control.

These are existing probability/statistical constructions, not a novelty claim.
Guarantees condition on fixed query positions and independent model draws, and
do not cover prediction misspecification, arbitrary online queries or a mission.
"""
import time
import numpy as np
from scipy.stats import beta, ncx2


def exact_isotropic_union(means, queries, sigma, radius):
    d2 = np.sum((queries[:, None, :]-means[None, :, :])**2, axis=-1)
    individual = ncx2.cdf(radius**2/sigma**2, 2, d2/sigma**2)
    return -np.expm1(np.log1p(-np.clip(individual, 0, 1)).sum(axis=-1))


def cp_interval(counts, n, alpha):
    """Two-sided exact binomial interval with total error alpha per query/look."""
    counts = np.asarray(counts, dtype=np.int64)
    lower = np.zeros(counts.shape, dtype=float)
    upper = np.ones(counts.shape, dtype=float)
    mask = counts > 0
    lower[mask] = beta.ppf(alpha/2, counts[mask], n-counts[mask]+1)
    mask = counts < n
    upper[mask] = beta.ppf(1-alpha/2, counts[mask]+1, n-counts[mask])
    assert np.isfinite(lower).all() and np.isfinite(upper).all()
    return lower, upper


def decisions(lower, upper, threshold):
    result = np.full(len(lower), 'unresolved', dtype=object)
    result[upper <= threshold] = 'safe'
    result[lower > threshold] = 'unsafe'
    return result


def event_counts(targets, queries, radius):
    # Common target draws across queries permit arbitrary cross-query dependence.
    # Independent draws along the sample dimension give Bernoulli observations.
    d2 = np.sum((targets[:, None, :, :]-queries[None, :, None, :])**2, axis=-1)
    return (d2 <= radius**2).any(axis=-1).sum(axis=0)


def fixed_event_mc(means, queries, sigma, radius, n, seed, threshold=.05, family_alpha=.05, guard=False):
    started = time.perf_counter()
    generator = np.random.default_rng(seed)
    counts = np.zeros(len(queries), dtype=np.int64)
    for offset in range(0, n, 512):
        size = min(512, n-offset)
        targets = generator.normal(size=(size, len(means), 2))*sigma+means
        counts += event_counts(targets, queries, radius)
    estimate = counts/n
    if guard:
        lower, upper = cp_interval(counts, n, family_alpha/len(queries))
        decision = decisions(lower, upper, threshold)
    else:
        lower, upper = np.full(len(queries), np.nan), np.full(len(queries), np.nan)
        decision = np.where(estimate <= threshold, 'safe', 'unsafe')
    return dict(estimate=estimate, lower=lower, upper=upper, decision=decision,
                counts=counts, samples=np.full(len(queries), n),
                pair_evaluations=n*len(means)*len(queries), target_vectors_drawn=n*len(means),
                elapsed_s=time.perf_counter()-started)


def sequential_event_mc(means, queries, sigma, radius, looks, seed, threshold=.05, family_alpha=.05):
    started = time.perf_counter()
    assert looks == sorted(set(looks)) and looks[0] > 0
    q = len(queries)
    generator = np.random.default_rng(seed)
    counts = np.zeros(q, dtype=np.int64)
    used = np.zeros(q, dtype=np.int64)
    lower, upper = np.zeros(q), np.ones(q)
    decision = np.full(q, 'unresolved', dtype=object)
    active = np.arange(q)
    previous = 0
    evaluations = 0
    # Bonferroni over all fixed queries and all predeclared looks. This permits
    # selecting any certified-safe candidate after observing these intervals.
    local_alpha = family_alpha/(q*len(looks))
    for n in looks:
        if not len(active):
            break
        for offset in range(previous, n, 512):
            size = min(512, n-offset)
            targets = generator.normal(size=(size, len(means), 2))*sigma+means
            counts[active] += event_counts(targets, queries[active], radius)
            evaluations += size*len(means)*len(active)
        used[active] = n
        lo, hi = cp_interval(counts[active], n, local_alpha)
        lower[active], upper[active] = lo, hi
        decision[active] = decisions(lo, hi, threshold)
        active = active[decision[active] == 'unresolved']
        previous = n
    return dict(estimate=counts/used, lower=lower, upper=upper, decision=decision,
                counts=counts, samples=used, pair_evaluations=evaluations,
                target_vectors_drawn=previous*len(means), elapsed_s=time.perf_counter()-started)


def self_check():
    # Zero/all successes must not produce degenerate zero-width confidence claims.
    lo, hi = cp_interval(np.array([0, 100]), 100, .01)
    assert lo[0] == 0 and hi[0] > 0 and lo[1] < 1 and hi[1] == 1
    means = np.array([[0., 0.], [.6, -.2]])
    query = np.array([[0., 0.], [2., 0.], [1., .2]])
    large = fixed_event_mc(means, query, .3, .825, 8192, 991)
    seq = sequential_event_mc(means, query, .3, .825, [512, 2048, 8192], 991)
    for j,n in enumerate(seq['samples']):
        prefix = fixed_event_mc(means, query, .3, .825, int(n), 991)
        assert seq['counts'][j] == prefix['counts'][j]
    assert np.max(abs(large['estimate']-exact_isotropic_union(means, query, .3, .825))) < .025
    # Repeated-look protection is stricter than a fixed-look interval at same n.
    n = 2048
    lo1,hi1=cp_interval(np.array([100]),n,.05/3)
    lo2,hi2=cp_interval(np.array([100]),n,.05/(3*3))
    assert lo2[0] <= lo1[0] and hi2[0] >= hi1[0]
    return dict(binomial_extremes='passed', shared_sample_prefixes='passed',
                event_reference_sanity='passed', multiple_look_alpha='passed')
