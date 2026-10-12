"""Diagnose small-sample predictive calibration; no manuscript outputs are written.

Two complementary experiments (fixed grid, paired comparisons):

* ``real`` reuses the v3 train/test/external splits and fitted classifier, and
  changes ONLY the uniform prior concentration. U(1) also reuses the original
  predictive random stream. No extra history is carved out of training.
* ``controlled`` freezes class-specific score probabilities from ONE reference
  fit, pooling its two held-out histograms. These probabilities, smoothed by
  ``template_mass`` TOTAL pseudo-counts per class (default .01), define a semi-synthetic DGP,
  NOT the unknown real population. Fresh test, external and historical counts
  are independently drawn from that DGP. Uniform, historical and oracle prior
  profiles are crossed with concentration and sample size. ``known_p`` samples
  the true predictive law directly: unlike an oracle PROFILE at finite m, it
  has no parameter uncertainty. Each reference fit is summarized separately.

Defaults: Fertility, Alzheimer, Sonar, Banknote; K=10; m=.1,1,5,20;
300 replications, 4000 predictive draws, history=500 per class; controlled
class counts = original counts times 1,4,16. This is a substantial experiment.
Small run first; do not draw scientific conclusions from --smoke.
The tiny DGP smoothing ensures a strictly positive exact oracle profile; it is
separate from the historical-profile smoother (one pseudo-count per class).
Vary --references and --template-mass to assess dependence on the frozen DGP.

From the repository root:
    py/.venv/bin/python -B py/run_verif.py --self-test
    py/.venv/bin/python -B py/run_verif.py --smoke --summary-only
    py/.venv/bin/python -B py/run_verif.py --arm real --summary-only
    py/.venv/bin/python -B py/run_verif.py --arm controlled --references 0 1 2

Results are saved by default to results/current_experiments/run_verif.csv,
relative to this script. Existing files are never overwritten. --output-csv
chooses a different destination; smoke runs default to run_verif_smoke.csv.
The record_type column distinguishes raw measurements, summaries and paired
contrasts. Metrics are numeric columns: a summary contains means and *_se;
a contrast contains method-minus-reference differences and paired *_se.
Metadata, histograms and template probabilities are preserved in details_json.
Coverage is a proportion, not a percentage. --summary-only omits data/raw rows;
--jsonl also prints the original JSON Lines. Progress goes to stderr.
The CSV is published only after completion and validation. Only cached,
checksum-verified datasets are accepted; no dataset is downloaded.

Interpretation: compare profiles AT THE SAME m, concentrations AT THE SAME
profile, and sizes within the SAME frozen DGP. Report coverage AND width,
paired differences and their MC standard errors. Larger coverage is not
automatically better calibration. All contrasts are exploratory; there is no
automatic selection of a 'best' prior. Real and controlled coverage estimates
have different sampling targets. Failure of a known-p control warrants checking
MC error/discreteness before interpreting the prior comparisons. The script
never asserts the desired scientific conclusion.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import platform
import struct
import sys
import tempfile
import unittest
from collections import Counter, defaultdict
from contextlib import redirect_stderr
from dataclasses import asdict, dataclass
from importlib.metadata import version
from itertools import product
from pathlib import Path
from unittest.mock import patch

# Also suppress imported-module bytecode when invoked without -B.
sys.dont_write_bytecode = True
if not __debug__:
    raise RuntimeError("Runtime assertions are required; do not use python -O.")

import numpy as np
from scipy.stats import multinomial
from threadpoolctl import threadpool_limits

import roc_utils.real_datasets as real_data
from roc_utils.evaluation_criteria import (
    is_external_auc_covered, is_external_roc_covered,
    summarize_mean, summarize_paired_differences,
)
from roc_utils.posterior_predictive_roc_sampler import (
    equal_tailed_interval, fixed_uniform_grid, grid_auc, roc_from_histograms,
    rocs_from_histogram_samples, simultaneous_euclidean_radii,
)
from roc_utils.prior_profiles import (
    dirichlet_parameters_from_bin_probabilities,
    expected_roc_from_bin_probabilities,
    historical_histogram_prior_bin_probabilities,
    validate_prior_mean_probabilities,
)
from roc_utils.semantic_rng import make_semantic_rng, semantic_stream_id
from roc_utils.utilities_experiments import (
    PredictiveBand, TestExternalHistograms, build_posterior_predictive_band,
    build_test_external_histograms, evaluate_replication,
)
from run_experiment_empirical_evaluation_on_real_world_datasets import (
    DEFAULT_MASTER_SEED, DEFAULT_OUTPUT_CSV, REAL_DATA_CONDITIONS,
    _posterior_stream_key, _prepare_dataset_replication,
)

PY_DIR = Path(__file__).resolve().parent
DEFAULT_VERIF_CSV = PY_DIR / "results" / "current_experiments" / "run_verif.csv"
LEVELS = (0.8, 0.95)
STREAM_NAMESPACE = 92901
PROFILE_IDS = {"uniform": 1, "historical": 2, "oracle": 3, "known_p": 4}
DEFAULT_DATASETS = ("Fertility", "Alzheimer", "Sonar", "Banknote Authentication")
HISTOGRAM_FIELDS = (
    "d_plus_test", "d_minus_test", "d_plus_external", "d_minus_external",
)


@dataclass(frozen=True)
class Config:
    arm: str = "both"
    datasets: tuple[str, ...] = DEFAULT_DATASETS
    replications: int = 300
    samples: int = 4000
    bins: int = 10
    masses: tuple[float, ...] = (0.1, 1.0, 5.0, 20.0)
    multipliers: tuple[int, ...] = (1, 4, 16)
    references: tuple[int, ...] = (0,)
    history_size: int = 500
    template_mass: float = 0.01
    seed: int = DEFAULT_MASTER_SEED
    summary_only: bool = False

    def __post_init__(self):
        assert self.arm in ("real", "controlled", "both")
        assert self.datasets and len(set(self.datasets)) == len(self.datasets)
        assert all(name in real_data.REAL_DATASET_NAMES for name in self.datasets)
        assert type(self.replications) is int and 2 <= self.replications < 1_000_000
        assert type(self.samples) is int and self.samples >= 40
        assert self.bins in (5, 10, 20)
        assert self.masses and 1.0 in self.masses, "U(1) is the paired baseline."
        assert len(set(self.masses)) == len(self.masses)
        # The shared converter checks sum(alpha)==m to absolute 1e-12.
        # At m=10000 harmless roundoff can violate that tolerance.
        assert all(np.isfinite(m) and 0 < m <= 1000 for m in self.masses), (
            "Supported experimental concentrations satisfy 0 < m <= 1000."
        )
        assert self.multipliers and len(set(self.multipliers)) == len(self.multipliers)
        assert all(type(s) is int and 1 <= s <= 1000 for s in self.multipliers)
        assert self.references and len(set(self.references)) == len(self.references)
        assert all(type(r) is int and 0 <= r < 1_000_000 for r in self.references)
        assert type(self.history_size) is int and self.history_size > 0
        assert np.isfinite(self.template_mass) and self.template_mass > 0
        assert type(self.seed) is int and 0 <= self.seed < 2**32


def emit(kind, **payload):
    print(json.dumps({"kind": kind, **payload}, allow_nan=False, sort_keys=True))


def freeze(values):
    array = np.asarray(values).copy()
    array.setflags(write=False)
    return array


def histogram_arrays(h):
    arrays = tuple(getattr(h, field) for field in HISTOGRAM_FIELDS)
    assert len({x.shape for x in arrays}) == 1
    for x in arrays:
        assert x.ndim == 1 and x.size >= 2
        assert np.issubdtype(x.dtype, np.integer)
        assert np.all(x >= 0) and x.sum() > 0
        assert not x.flags.writeable
    assert arrays[0].sum() == arrays[2].sum()
    assert arrays[1].sum() == arrays[3].sum()
    return arrays


def data_fingerprint(h):
    # Explicit byte order makes hashes independent of the host integer dtype.
    return hashlib.sha256(b"".join(
        np.asarray(x, dtype="<i8").tobytes() for x in histogram_arrays(h)
    )).hexdigest()


def prior_key(profile, mass):
    # Stable across loop reordering, selected subsets and Python processes.
    assert profile in PROFILE_IDS
    assert np.isfinite(mass) and mass >= 0
    return (PROFILE_IDS[profile], *struct.unpack(">II", struct.pack(">d", mass)))


def local_rng(config, *key):
    return make_semantic_rng(config.seed, STREAM_NAMESPACE, *key)


def cached_dataset(name):
    metadata = real_data.real_dataset_metadata(name)
    path = real_data.DEFAULT_REAL_DATA_DIR / f"{metadata.slug}.arff"
    if not path.is_file():
        raise FileNotFoundError(f"Cached source required (no download): {path}")
    assert hashlib.md5(path.read_bytes()).hexdigest() == metadata.md5_checksum

    def require_cached_source(spec, actual_path):
        assert actual_path == path and spec.md5_checksum == metadata.md5_checksum
        assert actual_path.is_file()
        assert hashlib.md5(actual_path.read_bytes()).hexdigest() == spec.md5_checksum

    # The repository loader is reused, but its acquisition hook is read-only.
    with patch.object(real_data, "_ensure_source_file", require_cached_source):
        loaded = real_data.load_real_dataset(name)
    assert loaded.n_complete_rows == metadata.n_complete_rows
    return loaded, metadata


def real_histograms(loaded, metadata, replication, config):
    index = real_data.REAL_DATASET_NAMES.index(metadata.name)
    prepared, scores, key = _prepare_dataset_replication(
        loaded, metadata, index, replication, master_seed=config.seed,
    )
    sets = [set(x.tolist()) for x in (
        prepared.train_row_indices, prepared.test_row_indices,
        prepared.external_row_indices,
    )]
    assert all(sets[i].isdisjoint(sets[j]) for i in range(3) for j in range(i))
    assert sum(map(len, sets)) == loaded.n_complete_rows
    h = build_test_external_histograms(
        scores.positive_test_scores, scores.negative_test_scores,
        scores.positive_external_scores, scores.negative_external_scores,
        fixed_uniform_grid(config.bins),
    )
    arrays = histogram_arrays(h)
    assert arrays[0].sum() == metadata.n_positive // 5
    assert arrays[1].sum() == metadata.n_negative // 5
    return h, {"fit_iterations": scores.n_iterations,
               "split_stream": semantic_stream_id(config.seed, *key)}


def template_probabilities(h, smoothing):
    assert np.isfinite(smoothing) and smoothing > 0
    dp, dm, ep, em = histogram_arrays(h)
    probabilities = tuple(freeze((c + smoothing / c.size) / (c.sum() + smoothing))
                          for c in (dp + ep, dm + em))
    validate_prior_mean_probabilities(*probabilities)
    return probabilities


def simulated_data(p, n, config, key):
    validate_prior_mean_probabilities(*p)
    assert len(n) == 2 and all(type(v) is int and v > 0 for v in n)
    # Separate semantic roles for test, external and history. No posterior RNG
    # can consume these streams, and changing m never changes these samples.
    test = local_rng(config, *key, 1)
    external = local_rng(config, *key, 2)
    historical = local_rng(config, *key, 3)
    counts = tuple(freeze(rng.multinomial(size, prob))
                   for rng in (test, external) for size, prob in zip(n, p))
    h = TestExternalHistograms(*counts)
    histogram_arrays(h)
    history = tuple(freeze(historical.multinomial(config.history_size, prob))
                    for prob in p)
    assert all(x.sum() == config.history_size for x in history)
    _, qp, qm = historical_histogram_prior_bin_probabilities(*history)
    return h, history, (freeze(qp), freeze(qm))


def known_probability_band(p, n, samples, rng):
    """Positive control: true future law, NOT an estimated/oracle prior."""
    fpr, tpr, _ = expected_roc_from_bin_probabilities(*p)
    draws = [rng.multinomial(size, prob, size=samples) for size, prob in zip(n, p)]
    for d, size in zip(draws, n):
        assert d.shape == (samples, p[0].size)
        assert np.all(d >= 0) and np.all(d.sum(axis=1) == size)
    fs, ts = rocs_from_histogram_samples(*draws)
    auc = grid_auc(fs, ts)
    return PredictiveBand(fpr, tpr,
                          simultaneous_euclidean_radii(fs, ts, fpr, tpr, LEVELS),
                          {level: equal_tailed_interval(auc, level) for level in LEVELS},
                          auc)


def auc_moments(mu, covariances):
    """Exact mean/variance for independent random class-proportion vectors."""
    pp, pm = mu
    cp, cm = covariances
    k = pp.size
    weights = np.tril(np.ones((k, k)), -1) + 0.5 * np.eye(k)
    mean = float(pp @ weights @ pm)
    hp, hm = weights @ pm, weights.T @ pp
    variance = float(hp @ cp @ hp + hm @ cm @ hm
                     + np.trace(weights.T @ cp @ weights @ cm))
    assert 0 <= mean <= 1 and -1e-12 <= variance <= 0.25 + 1e-12
    return mean, max(0.0, variance)


def prediction_moments(h, alpha=None, p=None):
    assert (alpha is None) != (p is None)
    observed = histogram_arrays(h)[:2]
    mu, covariance = [], []
    for index, counts in enumerate(observed):
        n = int(counts.sum())
        if alpha is not None:
            a = np.asarray(alpha[index])
            assert a.shape == counts.shape and np.all(a > 0)
            total = n + float(a.sum())
            probability = (counts + a) / total
            scale = (n + total) / (n * (total + 1))
        else:
            probability = p[index]
            scale = 1.0 / n
        assert np.isclose(probability.sum(), 1, atol=1e-12, rtol=0)
        mu.append(probability)
        covariance.append(scale * (np.diag(probability) - np.outer(probability, probability)))
    mean, variance = auc_moments(mu, covariance)
    return tuple(mu), mean, variance


def measurement_rows(h, band, mu, mean, variance):
    measurements = evaluate_replication(h, band)
    dp, dm, ep, em = histogram_arrays(h)
    test_auc = float(grid_auc(*roc_from_histograms(dp, dm)))
    center_auc = float(grid_auc(band.fpr_center, band.tpr_center))
    assert math.isclose(center_auc, mean, abs_tol=1e-12, rel_tol=0)
    common = {"test_auc": test_auc, "external_auc": measurements.external_grid_auc,
              "predictive_auc_mean": mean, "predictive_auc_variance": variance,
              "center_shift": mean - test_auc,
              "auc_error": measurements.external_grid_auc - mean,
              "auc_squared_error": (measurements.external_grid_auc - mean)**2}
    for label, test, ext, predicted in zip(("plus", "minus"), (dp, dm), (ep, em), mu):
        empty = test == 0
        common[f"empty_bins_{label}"] = int(empty.sum())
        common[f"unseen_predicted_{label}"] = float(predicted[empty].sum())
        common[f"unseen_external_{label}"] = float(ext[empty].sum() / ext.sum())
    rows = []
    for level in LEVELS:
        radius = measurements.roc_radii[level]
        lower, upper = measurements.auc_intervals[level]
        auc = measurements.external_grid_auc
        covered = is_external_auc_covered(auc, (lower, upper))
        below, above = auc < lower - 1e-12, auc > upper + 1e-12
        assert int(covered) + int(below) + int(above) == 1
        rows.append(dict(common, level=level, roc_radius=radius,
                         auc_lower=lower, auc_upper=upper, auc_width=upper - lower,
                         roc_deviation=measurements.external_max_roc_deviation,
                         roc_covered=int(is_external_roc_covered(
                             measurements.external_max_roc_deviation, radius)),
                         auc_covered=int(covered), auc_below=int(below), auc_above=int(above)))
    assert rows[1]["roc_radius"] >= rows[0]["roc_radius"]
    assert rows[1]["auc_lower"] <= rows[0]["auc_lower"]
    assert rows[1]["auc_upper"] >= rows[0]["auc_upper"]
    assert rows[1]["roc_covered"] >= rows[0]["roc_covered"]
    assert rows[1]["auc_covered"] >= rows[0]["auc_covered"]
    return rows


def evaluate_condition(h, profile, mass, q, rng, config, p=None):
    before = data_fingerprint(h)
    if profile == "known_p":
        assert mass == 0 and p is not None
        n = tuple(int(x.sum()) for x in histogram_arrays(h)[:2])
        band = known_probability_band(p, n, config.samples, rng)
        mu, mean, variance = prediction_moments(h, p=p)
    else:
        alpha = dirichlet_parameters_from_bin_probabilities(*q, mass)
        for a, probability in zip(alpha, q):
            assert np.isclose(a.sum(), mass, rtol=0, atol=1e-10)
            np.testing.assert_allclose(a / mass, probability, rtol=0, atol=1e-12)
        band = build_posterior_predictive_band(
            h.d_plus_test, h.d_minus_test, *alpha, config.samples, LEVELS, rng,
        )
        mu, mean, variance = prediction_moments(h, alpha=alpha)
    rows = measurement_rows(h, band, mu, mean, variance)
    assert data_fingerprint(h) == before, "A method modified the paired observations."
    return rows


SUMMARY_METRICS = (
    "roc_covered", "auc_covered", "roc_radius", "auc_width", "auc_below", "auc_above",
    "predictive_auc_mean", "predictive_auc_variance", "center_shift", "auc_error",
    "auc_squared_error", "unseen_predicted_plus", "unseen_external_plus",
    "unseen_predicted_minus", "unseen_external_minus",
)
CONTRAST_METRICS = ("roc_covered", "auc_covered", "roc_radius", "auc_width", "auc_error")
CONTROLLED_METRICS = (
    "unseen_true_plus", "unseen_true_minus", "unseen_error_plus", "unseen_error_minus",
    "predictive_auc_bias_to_truth",
)


def wilson_interval(indicators):
    """Descriptive 95% binomial interval, including all-success/all-failure runs."""
    values = np.asarray(indicators)
    assert values.ndim == 1 and values.size >= 2
    assert np.all((values == 0) | (values == 1))
    n, rate, z = values.size, float(values.mean()), 1.959963984540054
    denominator = 1 + z*z/n
    midpoint = (rate + z*z/(2*n)) / denominator
    halfwidth = z * math.sqrt(rate*(1-rate)/n + z*z/(4*n*n)) / denominator
    return [max(0.0, midpoint-halfwidth), min(1.0, midpoint+halfwidth)]


def paired_rows(left, right, metric):
    assert [x["replication"] for x in left] == [x["replication"] for x in right]
    assert [x["data_id"] for x in left] == [x["data_id"] for x in right]
    return asdict(summarize_paired_differences(
        [x[metric] for x in left], [x[metric] for x in right],
    ))


def summarize_group(records, context, config, report=None):
    report = emit if report is None else report
    groups = defaultdict(list)
    for row in records:
        groups[(row["profile"], row["mass"], row["level"])].append(row)
    for rows in groups.values():
        rows.sort(key=lambda row: row["replication"])
        assert [row["replication"] for row in rows] == list(range(config.replications))
    for key, rows in groups.items():
        profile, mass, level = key
        metrics = SUMMARY_METRICS + (CONTROLLED_METRICS if context["arm"] == "controlled" else ())
        summaries = {metric: asdict(summarize_mean([row[metric] for row in rows]))
                     for metric in metrics}
        for metric in ("roc_covered", "auc_covered"):
            summaries[metric]["coverage_minus_nominal"] = summaries[metric]["mean"] - level
            summaries[metric]["ci95_wilson"] = wilson_interval([row[metric] for row in rows])
        variance = summaries["predictive_auc_variance"]["mean"]
        report("summary", **context, profile=profile, mass=mass, level=level,
             metrics=summaries,
             mse_over_predictive_variance=(summaries["auc_squared_error"]["mean"] / variance
                                           if variance > 0 else None))
        # U1 is the main baseline; matched-m contrasts isolate profile effects;
        # matched-profile m1 contrasts isolate concentration effects.
        comparators = {("uniform", 1.0, level)}
        if profile in ("historical", "oracle"):
            comparators.add(("uniform", mass, level))
            comparators.add((profile, 1.0, level))
        if profile == "historical":
            comparators.add(("oracle", mass, level))
        for other in sorted(comparators - {key}):
            assert other in groups
            report("paired_contrast", **context, profile=profile, mass=mass, level=level,
                 reference_profile=other[0], reference_mass=other[1],
                 direction="method_minus_reference",
                 metrics={metric: paired_rows(rows, groups[other], metric)
                          for metric in CONTRAST_METRICS})


def original_baseline(config):
    """Optional read-only regression check against existing official rows."""
    if config.seed != DEFAULT_MASTER_SEED or not DEFAULT_OUTPUT_CSV.is_file():
        return {}
    result = {}
    with DEFAULT_OUTPUT_CSV.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if (row["dataset_name"] in config.datasets and int(row["K"]) == config.bins
                    and int(row["S"]) == config.samples
                    and int(row["replication"]) < config.replications):
                key = (row["dataset_name"], int(row["replication"]), float(row["level"]))
                assert key not in result
                result[key] = row
    return result


def check_original(row, original):
    for current, old in (("roc_radius", "roc_radius"), ("auc_lower", "auc_interval_lower"),
                         ("auc_upper", "auc_interval_upper"),
                         ("external_auc", "external_grid_auc"),
                         ("roc_deviation", "external_max_roc_deviation")):
        assert math.isclose(row[current], float(original[old]), rel_tol=0, abs_tol=2e-12), (
            f"U1 differs from official v3: {current}; inspect source/software versions."
        )
    for field in ("roc_covered", "auc_covered"):
        assert row[field] == int(original[field])


def run_real(loaded, metadata, config, originals, report=emit):
    index = real_data.REAL_DATASET_NAMES.index(metadata.name)
    condition = next(c for c in REAL_DATA_CONDITIONS
                     if c.dataset_index == index and c.n_bins == config.bins)
    q = (freeze(np.full(config.bins, 1 / config.bins)),) * 2
    context = dict(arm="real", dataset=metadata.name, K=config.bins,
                   n_plus=metadata.n_positive // 5, n_minus=metadata.n_negative // 5)
    records = []
    checked = 0
    for rep in range(config.replications):
        h, details = real_histograms(loaded, metadata, rep, config)
        data_id = data_fingerprint(h)
        if not config.summary_only:
            report("data", **context, replication=rep, data_id=data_id, **details,
                 histograms=[x.tolist() for x in histogram_arrays(h)])
        for mass in config.masses:
            key = (_posterior_stream_key(condition, rep) if mass == 1
                   else (STREAM_NAMESPACE, 1, index, config.bins, rep, *prior_key("uniform", mass)))
            rng = make_semantic_rng(config.seed, *key)
            for values in evaluate_condition(h, "uniform", mass, q, rng, config):
                row = dict(values, replication=rep, data_id=data_id, profile="uniform", mass=mass)
                original = originals.get((metadata.name, rep, row["level"]))
                if mass == 1 and original is not None:
                    check_original(row, original)
                    checked += 1
                records.append(row)
                if not config.summary_only:
                    report("raw", **context, **row,
                         simulation_stream=semantic_stream_id(config.seed, *key))
        if (rep + 1) % 25 == 0:
            print(f"real {metadata.name}: {rep + 1}/{config.replications}", file=sys.stderr)
    assert len(records) == config.replications * len(config.masses) * len(LEVELS)
    summarize_group(records, context, config, report)
    report("real_complete", **context, original_rows_verified=checked)


def run_controlled(loaded, metadata, config, report=emit):
    index = real_data.REAL_DATASET_NAMES.index(metadata.name)
    for reference in config.references:
        # Separate split IDs from all real-arm replication IDs.
        template_rep = 1_000_000 + reference
        h_reference, details = real_histograms(loaded, metadata, template_rep, config)
        p = template_probabilities(h_reference, config.template_mass)
        true_auc = expected_roc_from_bin_probabilities(*p)[2]
        report("template", dataset=metadata.name, reference=reference, K=config.bins,
             smoothing_total_per_class=config.template_mass, **details,
             reference_histograms=[x.tolist() for x in histogram_arrays(h_reference)],
             p_plus=p[0].tolist(), p_minus=p[1].tolist(),
             true_grid_auc=true_auc,
             interpretation="Frozen semi-synthetic DGP, not the real population.")
        for multiplier in config.multipliers:
            n = (metadata.n_positive // 5 * multiplier, metadata.n_negative // 5 * multiplier)
            context = dict(arm="controlled", dataset=metadata.name, K=config.bins,
                           reference=reference, multiplier=multiplier, n_plus=n[0], n_minus=n[1])
            records = []
            for rep in range(config.replications):
                key = (2, index, reference, config.bins, multiplier, rep)
                h, history, historical_q = simulated_data(p, n, config, key)
                data_id = data_fingerprint(h)
                uniform = (freeze(np.full(config.bins, 1 / config.bins)),) * 2
                profiles = {"uniform": uniform, "historical": historical_q, "oracle": p}
                if not config.summary_only:
                    report("data", **context, replication=rep, data_id=data_id,
                         histograms=[x.tolist() for x in histogram_arrays(h)],
                         history=[x.tolist() for x in history],
                         data_streams=[semantic_stream_id(config.seed, STREAM_NAMESPACE, *key, role)
                                       for role in (1, 2, 3)])
                conditions = [(name, mass) for name in profiles for mass in config.masses]
                conditions.append(("known_p", 0.0))
                for profile, mass in conditions:
                    simulation_key = (*key, 4, *prior_key(profile, mass))
                    rng = local_rng(config, *simulation_key)
                    for values in evaluate_condition(
                        h, profile, mass, profiles.get(profile), rng, config, p=p,
                    ):
                        row = dict(values, replication=rep, data_id=data_id, profile=profile, mass=mass)
                        # Exact missing-bin mass under the known DGP removes
                        # external-sample noise from this particular diagnostic.
                        for label, counts, truth in zip(("plus", "minus"), histogram_arrays(h)[:2], p):
                            row[f"unseen_true_{label}"] = float(truth[counts == 0].sum())
                            row[f"unseen_error_{label}"] = (
                                row[f"unseen_predicted_{label}"] - row[f"unseen_true_{label}"])
                        row["predictive_auc_bias_to_truth"] = (
                            row["predictive_auc_mean"] - true_auc)
                        records.append(row)
                        if not config.summary_only:
                            report("raw", **context, **row,
                                 simulation_stream=semantic_stream_id(
                                     config.seed, STREAM_NAMESPACE, *simulation_key))
                if (rep + 1) % 25 == 0:
                    print(f"controlled {metadata.name} ref={reference} x{multiplier}: "
                          f"{rep + 1}/{config.replications}", file=sys.stderr)
            assert len(records) == config.replications * (3 * len(config.masses) + 1) * len(LEVELS)
            summarize_group(records, context, config, report)


def source_hashes():
    paths = [Path(__file__), PY_DIR / "run_experiment_empirical_evaluation_on_real_world_datasets.py",
             *sorted((PY_DIR / "roc_utils").glob("*.py"))]
    return {str(path.relative_to(PY_DIR)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in paths}


def run(config, report=emit):
    hashes = source_hashes()
    loaded = {name: cached_dataset(name) for name in config.datasets}
    report("metadata", schema="verif_v2", config=asdict(config), source_sha256=hashes,
         software={"python": platform.python_version(),
                   **{name: version(name) for name in ("numpy", "scipy", "scikit-learn")}},
         nominal_levels=LEVELS, se_scope="Conditional on corpus or frozen DGP; includes predictive MC.",
         interpretation="Exploratory factorial diagnostics; no causal conclusion is imposed.")
    originals = original_baseline(config) if config.arm in ("real", "both") else {}
    with threadpool_limits(limits=1):
        for name, (data, metadata) in loaded.items():
            print(f"Starting {name}, arm={config.arm}", file=sys.stderr)
            if config.arm in ("real", "both"):
                run_real(data, metadata, config, originals, report)
            if config.arm in ("controlled", "both"):
                run_controlled(data, metadata, config, report)
    assert source_hashes() == hashes, "Source files changed during the experiment."
    report("complete", schema="verif_v2", datasets=config.datasets, config=asdict(config))


CSV_METRICS = tuple(dict.fromkeys((*SUMMARY_METRICS, *CONTROLLED_METRICS,
    "test_auc", "external_auc", "auc_lower", "auc_upper", "roc_deviation",
    "empty_bins_plus", "empty_bins_minus")))
CSV_FIELDS = (
    "record_type", "schema", "arm", "dataset", "K", "reference", "multiplier",
    "n_plus", "n_minus", "replication", "data_id", "profile", "mass", "level",
    "reference_profile", "reference_mass", "direction", "B", "S", "seed",
    "history_size", "template_mass", "original_rows_verified", "fit_iterations",
    "split_stream", "simulation_stream", "mse_over_predictive_variance",
    *CSV_METRICS, *(f"{metric}_se" for metric in (*SUMMARY_METRICS, *CONTROLLED_METRICS)),
    "roc_covered_minus_nominal", "roc_covered_ci95_lower", "roc_covered_ci95_upper",
    "auc_covered_minus_nominal", "auc_covered_ci95_lower", "auc_covered_ci95_upper",
    "details_json",
)
assert len(CSV_FIELDS) == len(set(CSV_FIELDS))


def expected_csv_counts(config):
    datasets, masses, levels = len(config.datasets), len(config.masses), len(LEVELS)
    real_groups = datasets if config.arm in ("real", "both") else 0
    templates = datasets * len(config.references) if config.arm in ("controlled", "both") else 0
    controlled_groups = templates * len(config.multipliers)
    summaries = levels * (real_groups * masses + controlled_groups * (3*masses + 1))
    return Counter({
        "metadata": 1, "complete": 1, "template": templates, "real_complete": real_groups,
        "summary": summaries,
        "paired_contrast": levels * (real_groups*(masses-1) + controlled_groups*(8*masses-4)),
        "data": 0 if config.summary_only else config.replications*(real_groups + controlled_groups),
        "raw": 0 if config.summary_only else config.replications*summaries,
    })


def csv_record(kind, payload, config):
    """Flatten numeric measurements; retain remaining provenance without loss."""
    # Reject NaN/Infinity before either serializing numeric cells or metadata.
    json.dumps(payload, allow_nan=False)
    row = dict(record_type=kind, schema="verif_v2", B=config.replications,
               S=config.samples, seed=config.seed, history_size=config.history_size,
               template_mass=config.template_mass)
    details = {}
    for key, value in payload.items():
        if key == "metrics":
            for metric, estimate in value.items():
                assert metric in CSV_METRICS
                assert estimate["n_replications"] == config.replications
                row[metric] = estimate["mean"]
                row[f"{metric}_se"] = estimate["standard_error"]
                if "coverage_minus_nominal" in estimate:
                    row[f"{metric}_minus_nominal"] = estimate["coverage_minus_nominal"]
                    row[f"{metric}_ci95_lower"], row[f"{metric}_ci95_upper"] = estimate["ci95_wilson"]
        elif key in CSV_FIELDS:
            assert not isinstance(value, (dict, list, tuple)), f"Nested CSV value: {key}"
            row[key] = value
        else:
            details[key] = value
    row["details_json"] = json.dumps(details, sort_keys=True, allow_nan=False)
    assert set(row).issubset(CSV_FIELDS)
    return row


def validate_csv(path, config):
    """Read the entire finished CSV; check cardinality, identities and coverage."""
    counts, unique, paired = Counter(), set(), {}
    first = last = None
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == list(CSV_FIELDS)
        for row in reader:
            assert None not in row and all(value is not None for value in row.values())
            kind = row["record_type"]
            first = kind if first is None else first
            last = kind
            counts[kind] += 1
            assert row["schema"] == "verif_v2"
            assert int(row["B"]) == config.replications and int(row["S"]) == config.samples
            assert int(row["seed"]) == config.seed
            details = json.loads(row["details_json"])
            assert isinstance(details, dict)
            if kind in ("metadata", "complete"):
                assert details["config"] == json.loads(json.dumps(asdict(config)))
            if kind in ("raw", "summary", "paired_contrast"):
                assert row["dataset"] in config.datasets
                assert int(row["K"]) == config.bins and float(row["level"]) in LEVELS
                assert row["profile"] in PROFILE_IDS
                mass = float(row["mass"])
                assert mass == 0 if row["profile"] == "known_p" else mass in config.masses
                if row["arm"] == "controlled":
                    assert int(row["reference"]) in config.references
                    assert int(row["multiplier"]) in config.multipliers
                else:
                    assert row["arm"] == "real" and row["profile"] == "uniform"
                identity = tuple(row[key] for key in (
                    "record_type", "arm", "dataset", "K", "reference", "multiplier",
                    "profile", "mass", "level", "replication", "reference_profile", "reference_mass"))
                assert identity not in unique, f"Duplicate CSV result: {identity}"
                unique.add(identity)
                for metric in CSV_METRICS:
                    if row[metric]:
                        assert math.isfinite(float(row[metric]))
                    se = row.get(f"{metric}_se")
                    if se:
                        assert math.isfinite(float(se)) and float(se) >= 0
            if kind == "raw":
                assert 0 <= int(row["replication"]) < config.replications
                pair_key = tuple(row[key] for key in (
                    "arm", "dataset", "K", "reference", "multiplier", "replication"))
                assert paired.setdefault(pair_key, row["data_id"]) == row["data_id"]
                lower, upper = float(row["auc_lower"]), float(row["auc_upper"])
                assert 0 <= lower <= upper <= 1
                assert math.isclose(float(row["auc_width"]), upper-lower, abs_tol=1e-12)
                assert int(row["auc_covered"]) == is_external_auc_covered(float(row["external_auc"]), (lower, upper))
                assert int(row["roc_covered"]) == is_external_roc_covered(float(row["roc_deviation"]), float(row["roc_radius"]))
                assert sum(int(row[key]) for key in ("auc_covered", "auc_below", "auc_above")) == 1
            elif kind == "summary":
                for metric in ("auc_covered", "roc_covered"):
                    rate = float(row[metric])
                    assert 0 <= rate <= 1
                    assert float(row[f"{metric}_ci95_lower"]) <= rate + 1e-12
                    assert float(row[f"{metric}_ci95_upper"]) >= rate - 1e-12
                assert math.isclose(sum(float(row[key]) for key in
                    ("auc_covered", "auc_below", "auc_above")), 1, abs_tol=1e-12)
    assert first == "metadata" and last == "complete", "Incomplete CSV."
    assert counts == expected_csv_counts(config), (counts, expected_csv_counts(config))
    return counts


def run_to_csv(config, destination, *, jsonl=False):
    """Publish a complete, validated file atomically without replacing any file."""
    destination = Path(destination).resolve()
    assert destination.suffix.lower() == ".csv", "The output must have a .csv extension."
    if destination.exists():
        raise FileExistsError(f"Output already exists; choose another --output-csv: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{destination.stem}-", suffix=".partial",
                                                 dir=destination.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
            writer.writeheader()

            def report(kind, **payload):
                writer.writerow(csv_record(kind, payload, config))
                if jsonl:
                    emit(kind, **payload)

            run(config, report)
            handle.flush()
            os.fsync(handle.fileno())
        counts = validate_csv(temporary, config)
        # Same-directory hard link atomically refuses a competing destination.
        os.link(temporary, destination)
        print(f"Validated CSV: {destination} ({sum(counts.values())} records)", file=sys.stderr)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


class VerificationTests(unittest.TestCase):
    """Tests live here because this task may change only run_verif.py."""

    @staticmethod
    def toy_histograms():
        return TestExternalHistograms(*(freeze(x) for x in ([0, 2], [3, 0], [1, 1], [2, 1])))

    def test_invalid_configurations_fail(self):
        for kwargs in ({"replications": 1}, {"masses": (5.0,)}, {"masses": (1.0, -1.0)},
                       {"masses": (1.0, 10000.0)},
                       {"template_mass": 0}, {"references": (0, 0)}, {"samples": 0}):
            with self.subTest(kwargs=kwargs), self.assertRaises(AssertionError):
                Config(**kwargs)

    def test_template_smooths_actual_dgp_and_keeps_class_totals_separate(self):
        p = template_probabilities(self.toy_histograms(), 1)
        np.testing.assert_allclose(p[0], [1.5 / 5, 3.5 / 5], atol=1e-15)
        np.testing.assert_allclose(p[1], [5.5 / 7, 1.5 / 7], atol=1e-15)

    def test_stream_roles_and_methods_do_not_change_paired_data(self):
        config = Config(replications=2, samples=80)
        p = (np.array([0.3, 0.7]), np.array([0.8, 0.2]))
        h1, history1, q1 = simulated_data(p, (2, 3), config, (2, 0, 0))
        local_rng(config, 99, *prior_key("uniform", 1)).random(100)
        h2, history2, q2 = simulated_data(p, (2, 3), config, (2, 0, 0))
        self.assertEqual(data_fingerprint(h1), data_fingerprint(h2))
        for a, b in zip((*history1, *q1), (*history2, *q2)):
            np.testing.assert_array_equal(a, b)
        keys = {prior_key(profile, mass) for profile in PROFILE_IDS for mass in config.masses}
        self.assertEqual(len(keys), len(PROFILE_IDS) * len(config.masses))

    def test_predictive_auc_moments_against_exhaustive_distribution(self):
        h = self.toy_histograms()
        alpha = (np.array([0.2, 0.8]), np.array([0.7, 0.3]))
        from scipy.stats import dirichlet_multinomial
        values, probabilities = [], []
        for i, j in product(range(3), range(4)):
            dp, dm = np.array([i, 2-i]), np.array([j, 3-j])
            probability = (dirichlet_multinomial.pmf(dp, h.d_plus_test + alpha[0], 2)
                           * dirichlet_multinomial.pmf(dm, h.d_minus_test + alpha[1], 3))
            probabilities.append(probability)
            values.append(float(grid_auc(*roc_from_histograms(dp, dm))))
        probabilities, values = np.array(probabilities), np.array(values)
        self.assertAlmostEqual(float(probabilities.sum()), 1)
        _, mean, variance = prediction_moments(h, alpha=alpha)
        self.assertAlmostEqual(mean, float(probabilities @ values))
        self.assertAlmostEqual(variance, float(probabilities @ ((values - mean)**2)))

    def test_oracle_missing_mass_identity_by_enumeration(self):
        p, n, mass = np.array([0.2, 0.8]), 3, 5.0
        actual, predicted = 0.0, 0.0
        for i in range(n + 1):
            counts = np.array([i, n - i])
            weight = multinomial.pmf(counts, n, p)
            empty = counts == 0
            actual += weight * p[empty].sum()
            predicted += weight * ((counts + mass*p)/(n+mass))[empty].sum()
        self.assertAlmostEqual(actual, float(np.sum(p * (1-p)**n)))
        self.assertAlmostEqual(predicted, mass/(n+mass)*actual)

    def test_known_p_control_has_no_parameter_uncertainty(self):
        p = (np.array([0.3, 0.7]), np.array([0.8, 0.2]))
        h = self.toy_histograms()
        _, mean, variance = prediction_moments(h, p=p)
        vals, probs = [], []
        for i, j in product(range(3), range(4)):
            dp, dm = np.array([i, 2-i]), np.array([j, 3-j])
            vals.append(float(grid_auc(*roc_from_histograms(dp, dm))))
            probs.append(multinomial.pmf(dp, 2, p[0])*multinomial.pmf(dm, 3, p[1]))
        self.assertAlmostEqual(mean, float(np.dot(vals, probs)))
        self.assertAlmostEqual(variance, float(np.dot((np.array(vals)-mean)**2, probs)))
        band = known_probability_band(p, (2, 3), 100, np.random.default_rng(123))
        measurement_rows(h, band, p, mean, variance)

    def test_paired_standard_error_and_pairing_guards(self):
        a = [dict(replication=i, data_id=str(i), value=v) for i, v in enumerate([1, 0, 1, 0])]
        b = [dict(replication=i, data_id=str(i), value=v) for i, v in enumerate([0, 1, 0, 0])]
        result = paired_rows(a, b, "value")
        self.assertAlmostEqual(result["mean"], 0.25)
        self.assertAlmostEqual(result["standard_error"], np.std([1, -1, 1, 0], ddof=1)/2)
        with self.assertRaises(AssertionError):
            paired_rows(a, b[::-1], "value")

    def test_extreme_coverage_does_not_have_zero_width_confidence_interval(self):
        lower, upper = wilson_interval([1] * 10)
        self.assertAlmostEqual(upper, 1)
        self.assertLess(lower, 0.8)
        lower, upper = wilson_interval([0] * 10)
        self.assertAlmostEqual(lower, 0)
        self.assertGreater(upper, 0.2)

    def test_controlled_summary_preserves_truth_and_historical_oracle_pairing(self):
        config = Config(replications=2, samples=80, masses=(1.0,))
        context = {"arm": "controlled"}
        h = self.toy_histograms()
        p = (np.array([0.3, 0.7]), np.array([0.8, 0.2]))
        records = []
        for rep in range(2):
            for profile in ("uniform", "historical", "oracle"):
                for values in evaluate_condition(h, profile, 1, p, local_rng(config, rep), config):
                    records.append(dict(values, replication=rep, profile=profile, mass=1.0,
                                        data_id=str(rep), **dict.fromkeys(CONTROLLED_METRICS, 0.0)))
        with patch(__name__ + ".emit") as output:
            summarize_group(records[::-1], context, config)
        summaries = [call.kwargs for call in output.call_args_list if call.args[0] == "summary"]
        self.assertEqual(len(summaries), 6)
        self.assertTrue(all("unseen_true_plus" in row["metrics"] for row in summaries))
        contrasts = [call.kwargs for call in output.call_args_list if call.args[0] == "paired_contrast"]
        self.assertEqual(sum(row["profile"] == "historical" and row["reference_profile"] == "oracle"
                             for row in contrasts), 2)

    def test_external_outcome_cannot_change_constructed_prediction(self):
        config = Config(replications=2, samples=100)
        h = self.toy_histograms()
        changed = TestExternalHistograms(h.d_plus_test, h.d_minus_test,
                                         freeze([2, 0]), freeze([0, 3]))
        q = (np.array([0.5, 0.5]),) * 2
        a = evaluate_condition(h, "uniform", 1, q, local_rng(config, 1), config)
        b = evaluate_condition(changed, "uniform", 1, q, local_rng(config, 1), config)
        for row_a, row_b in zip(a, b):
            for field in ("roc_radius", "auc_lower", "auc_upper", "predictive_auc_mean"):
                self.assertEqual(row_a[field], row_b[field])
        self.assertNotEqual(a[0]["external_auc"], b[0]["external_auc"])

    @staticmethod
    def tiny_csv(destination, summary_only=False):
        config = Config(datasets=("Fertility",), replications=2, samples=80,
                        masses=(1.0, 5.0), multipliers=(1,), seed=117,
                        summary_only=summary_only)
        metadata = real_data.real_dataset_metadata("Fertility")
        arrays = ([0, 2], [17, 0], [1, 1], [16, 1])
        h = TestExternalHistograms(*(freeze([*x, *([0]*8)]) for x in arrays))
        details = {"fit_iterations": 0, "split_stream": "test_fixture"}
        with patch(__name__ + ".cached_dataset", return_value=(object(), metadata)), \
             patch(__name__ + ".real_histograms", return_value=(h, details)), \
             redirect_stderr(io.StringIO()):
            run_to_csv(config, destination)
        return config

    def test_csv_round_trip_summaries_and_paired_contrasts(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "check.csv"
            config = self.tiny_csv(path)
            self.assertEqual(validate_csv(path, config), expected_csv_counts(config))
            with path.open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            raw = defaultdict(list)
            key_fields = ("arm", "dataset", "K", "reference", "multiplier", "profile", "mass", "level")
            for row in rows:
                if row["record_type"] == "raw":
                    raw[tuple(row[k] for k in key_fields)].append(row)
            for row in rows:
                if row["record_type"] not in ("summary", "paired_contrast"):
                    continue
                key = tuple(row[k] for k in key_fields)
                values = np.array([float(r["auc_covered"]) for r in raw[key]])
                if row["record_type"] == "paired_contrast":
                    reference = dict(row, profile=row["reference_profile"], mass=row["reference_mass"])
                    other = raw[tuple(reference[k] for k in key_fields)]
                    values -= [float(r["auc_covered"]) for r in other]
                self.assertAlmostEqual(float(row["auc_covered"]), float(values.mean()))
                self.assertAlmostEqual(float(row["auc_covered_se"]), float(values.std(ddof=1)/math.sqrt(2)))
            summaries = [r for r in rows if r["record_type"] == "summary" and r["arm"] == "controlled"]
            self.assertTrue(all(r["unseen_true_plus"] != "" for r in summaries))
            self.assertEqual(len(list(Path(directory).iterdir())), 1)

    def test_summary_only_csv_is_valid_and_metadata_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "summary.csv"
            config = self.tiny_csv(path, summary_only=True)
            counts = validate_csv(path, config)
            self.assertEqual(counts["raw"], 0)
            with path.open(newline="") as handle:
                first = next(csv.DictReader(handle))
            metadata = json.loads(first["details_json"])
            self.assertIn("source_sha256", metadata)
            self.assertEqual(metadata["config"]["seed"], config.seed)

    def test_csv_never_overwrites_or_publishes_an_incomplete_run(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "existing.csv"
            path.write_text("previous results", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                run_to_csv(Config(), path)
            self.assertEqual(path.read_text(), "previous results")
            failed = Path(directory) / "failed.csv"
            with patch(__name__ + ".run", side_effect=RuntimeError("simulated failure")), \
                 self.assertRaisesRegex(RuntimeError, "simulated failure"):
                run_to_csv(Config(), failed)
            self.assertFalse(failed.exists())
            with patch(__name__ + ".run"), self.assertRaises(AssertionError):
                run_to_csv(Config(), failed)
            self.assertFalse(failed.exists())
            self.assertEqual(list(Path(directory).iterdir()), [path])

    def test_csv_rejects_nan_and_corrupted_coverage(self):
        with self.assertRaises(ValueError):
            csv_record("raw", {"external_auc": float("nan")}, Config())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "corrupt.csv"
            config = self.tiny_csv(path)
            with path.open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            row = next(r for r in rows if r["record_type"] == "raw")
            row["auc_covered"] = str(1-int(row["auc_covered"]))
            with path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaises(AssertionError):
                validate_csv(path, config)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--self-test", action="store_true", help="Run embedded tests only; no dataset needed.")
    parser.add_argument("--smoke", action="store_true", help="Tiny execution check; overrides B/S/reference/size budgets.")
    parser.add_argument("--arm", choices=("real", "controlled", "both"), default="both")
    parser.add_argument("--datasets", nargs="+", help="Registered names or slugs; quote names containing spaces.")
    parser.add_argument("--replications", type=int, default=300)
    parser.add_argument("--samples", type=int, default=4000)
    parser.add_argument("--bins", type=int, choices=(5, 10, 20), default=10)
    parser.add_argument("--masses", nargs="+", type=float, default=[0.1, 1, 5, 20])
    parser.add_argument("--multipliers", nargs="+", type=int, default=[1, 4, 16])
    parser.add_argument("--references", nargs="+", type=int, default=[0])
    parser.add_argument("--history-size", type=int, default=500)
    parser.add_argument("--template-mass", type=float, default=0.01,
                        help="DGP smoothing only; distinct from historical smoothing and prior concentration.")
    parser.add_argument("--seed", type=int, default=DEFAULT_MASTER_SEED)
    parser.add_argument("--summary-only", action="store_true", help="Omit per-replication data/raw records.")
    parser.add_argument("--output-csv", type=Path,
                        help=f"Destination CSV; default {DEFAULT_VERIF_CSV}. Existing files are protected.")
    parser.add_argument("--jsonl", action="store_true", help="Also print JSON Lines to stdout.")
    args = parser.parse_args(argv)
    if args.self_test:
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(VerificationTests)
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        return 0 if result.wasSuccessful() else 1
    names = args.datasets or (["Fertility"] if args.smoke else DEFAULT_DATASETS)
    config = Config(
        arm=args.arm, datasets=tuple(real_data.real_dataset_metadata(x).name for x in names),
        replications=3 if args.smoke else args.replications,
        samples=128 if args.smoke else args.samples, bins=args.bins,
        masses=tuple(args.masses), multipliers=(1, 4) if args.smoke else tuple(args.multipliers),
        references=(0,) if args.smoke else tuple(args.references),
        history_size=args.history_size, template_mass=args.template_mass,
        seed=args.seed, summary_only=args.summary_only,
    )
    if args.smoke:
        print("SMOKE CHECK ONLY: these budgets cannot support scientific conclusions.", file=sys.stderr)
    destination = args.output_csv or (DEFAULT_VERIF_CSV.with_name("run_verif_smoke.csv")
                                     if args.smoke else DEFAULT_VERIF_CSV)
    run_to_csv(config, destination, jsonl=args.jsonl)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
