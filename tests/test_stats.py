"""
Minimal test suite for core statistical functions used in the paper's analysis.

Run with: uv run pytest tests/test_stats.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Both analysis modules use relative imports (from models import ...), so we
# add the analysis directory to sys.path before importing.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "analysis"))

from verify_paper_stats import pearson_with_ci, within_country_pearson
from style import attack_mass, fit_logistic, join_col, logistic


# ── pearson_with_ci ────────────────────────────────────────────────────────────


class TestPearsonWithCI:
    """Tests for pearson_with_ci(x, y, alpha=0.05)."""

    def test_perfect_positive_correlation(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = pearson_with_ci(x, x)
        assert result["r"] == pytest.approx(1.0, abs=1e-4)
        assert result["p"] == pytest.approx(0.0, abs=1e-6)
        assert result["ci_lo"] == pytest.approx(1.0, abs=1e-4)
        assert result["ci_hi"] == pytest.approx(1.0, abs=1e-4)
        assert result["n"] == 5

    def test_perfect_negative_correlation(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = -x
        result = pearson_with_ci(x, y)
        assert result["r"] == pytest.approx(-1.0, abs=1e-4)
        assert result["p"] == pytest.approx(0.0, abs=1e-6)
        assert result["ci_lo"] == pytest.approx(-1.0, abs=1e-4)
        assert result["ci_hi"] == pytest.approx(-1.0, abs=1e-4)
        assert result["n"] == 5

    def test_zero_correlation(self):
        # x is monotone; y is symmetric around the midpoint — exactly zero Pearson r
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([1.0, -1.0, 0.0, -1.0, 1.0])
        result = pearson_with_ci(x, y)
        assert result["r"] == pytest.approx(0.0, abs=1e-4)
        assert result["p"] == pytest.approx(1.0, abs=1e-6)
        # CI must straddle zero and be symmetric for r=0
        assert result["ci_lo"] < 0.0
        assert result["ci_hi"] > 0.0
        assert result["ci_lo"] == pytest.approx(-result["ci_hi"], abs=1e-4)

    def test_ci_ordering_positive_r(self):
        # With a genuine positive correlation, ci_lo < r < ci_hi
        x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
        y = np.array([2, 4, 5, 4, 5, 7, 8, 9, 10, 12], dtype=float)
        result = pearson_with_ci(x, y)
        assert result["ci_lo"] < result["r"] < result["ci_hi"]
        assert result["ci_lo"] > 0.0   # strongly positive

    def test_ci_ordering_negative_r(self):
        x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
        y = np.array([12, 10, 9, 8, 7, 5, 4, 5, 2, 1], dtype=float)
        result = pearson_with_ci(x, y)
        assert result["ci_lo"] < result["r"] < result["ci_hi"]
        assert result["ci_hi"] < 0.0   # strongly negative

    def test_nan_values_are_excluded(self):
        # Inserting a NaN pair should give the same result as excluding it
        x_full = np.array([1.0, 2.0, 4.0, 5.0])
        y_full = np.array([2.0, 4.0, 8.0, 10.0])
        x_nan = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        y_nan = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
        r_full = pearson_with_ci(x_full, y_full)
        r_nan = pearson_with_ci(x_nan, y_nan)
        assert r_nan["r"] == pytest.approx(r_full["r"], abs=1e-4)
        assert r_nan["n"] == 4

    def test_insufficient_data_returns_nan(self):
        result = pearson_with_ci([1.0, 2.0], [3.0, 4.0])
        assert np.isnan(result["r"])
        assert np.isnan(result["p"])
        assert np.isnan(result["ci_lo"])
        assert np.isnan(result["ci_hi"])
        assert result["n"] == 2

    def test_narrower_ci_with_more_data(self):
        # More data → tighter CI for the same correlation structure
        rng = np.random.default_rng(0)
        x_small = rng.normal(0, 1, 10)
        y_small = x_small + rng.normal(0, 0.5, 10)
        x_large = rng.normal(0, 1, 200)
        y_large = x_large + rng.normal(0, 0.5, 200)
        r_small = pearson_with_ci(x_small, y_small)
        r_large = pearson_with_ci(x_large, y_large)
        width_small = r_small["ci_hi"] - r_small["ci_lo"]
        width_large = r_large["ci_hi"] - r_large["ci_lo"]
        assert width_large < width_small


# ── within_country_pearson ─────────────────────────────────────────────────────


class TestWithinCountryPearson:
    """Tests for within_country_pearson(df, xcol, ycol, group_col, alpha)."""

    def _make_df(self, rng_seed=0, signal=True):
        """Create a synthetic DataFrame with 3 countries × 10 periods."""
        rng = np.random.default_rng(rng_seed)
        rows = []
        for i, country in enumerate(["A", "B", "C"]):
            base = i * 0.4   # different mean theta per country
            for _ in range(10):
                theta = base + rng.normal(0, 0.1)
                if signal:
                    join = 1.0 - theta + rng.normal(0, 0.05)
                else:
                    join = rng.uniform(0, 1)
                rows.append({"country": country, "theta": theta, "jf": join})
        return pd.DataFrame(rows)

    def test_strong_within_signal_detected(self):
        df = self._make_df(signal=True)
        result = within_country_pearson(df, "theta", "jf", group_col="country")
        assert result["r"] < -0.5   # strong negative within-country relationship
        assert result["p"] < 0.01

    def test_no_within_signal_not_significant(self):
        df = self._make_df(signal=False)
        result = within_country_pearson(df, "theta", "jf", group_col="country")
        # Not expected to be significant at 1% level with pure noise
        assert abs(result["r"]) < 0.6

    def test_result_has_required_keys(self):
        df = self._make_df()
        result = within_country_pearson(df, "theta", "jf", group_col="country")
        for key in ("r", "p", "ci_lo", "ci_hi", "n"):
            assert key in result

    def test_insufficient_data_returns_nan(self):
        df = pd.DataFrame({"country": ["A", "B"], "theta": [0.1, 0.2], "jf": [0.5, 0.6]})
        result = within_country_pearson(df, "theta", "jf")
        assert np.isnan(result["r"])


# ── attack_mass ────────────────────────────────────────────────────────────────


class TestAttackMass:
    """Tests for attack_mass(theta, theta_star=0.50, sigma=0.30).

    Morris-Shin formula: x* = theta* + sigma*Phi_inv(theta*),
    A(theta) = Phi((x* - theta) / sigma).
    """

    def test_at_theta_star_equals_half(self):
        # When theta = theta_star = 0.5 and sigma = 0.3:
        # x* = 0.5 + 0.3*Phi_inv(0.5) = 0.5 + 0 = 0.5
        # A = Phi(0) = 0.5
        result = attack_mass(0.50, theta_star=0.50, sigma=0.30)
        assert result == pytest.approx(0.5, abs=1e-10)

    def test_at_theta_zero_near_one(self):
        # theta far below threshold: almost everyone attacks
        result = attack_mass(0.0, theta_star=0.50, sigma=0.30)
        assert result == pytest.approx(0.9522, abs=1e-3)
        assert result > 0.9

    def test_at_theta_one_near_zero(self):
        # theta far above threshold: almost no one attacks
        result = attack_mass(1.0, theta_star=0.50, sigma=0.30)
        assert result == pytest.approx(0.0478, abs=1e-3)
        assert result < 0.1

    def test_monotone_decreasing(self):
        # A(theta) must be strictly decreasing in theta
        thetas = np.linspace(0.0, 1.0, 20)
        values = attack_mass(thetas, theta_star=0.50, sigma=0.30)
        diffs = np.diff(values)
        assert np.all(diffs < 0), "attack_mass must be strictly decreasing in theta"

    def test_bounded_zero_one(self):
        thetas = np.linspace(-1.0, 2.0, 50)
        values = attack_mass(thetas, theta_star=0.50, sigma=0.30)
        assert np.all(values >= 0.0)
        assert np.all(values <= 1.0)

    def test_self_consistency_at_theta_star(self):
        # For any (theta_star, sigma), A(theta_star, theta_star, sigma) = theta_star
        # This follows from Phi((x* - theta*)/sigma) = Phi(Phi_inv(theta*)) = theta*
        for ts in [0.2, 0.3, 0.5, 0.7, 0.8]:
            result = attack_mass(ts, theta_star=ts, sigma=0.30)
            assert result == pytest.approx(ts, abs=1e-8), f"failed at theta_star={ts}"

    def test_array_input(self):
        thetas = np.array([0.0, 0.5, 1.0])
        results = attack_mass(thetas, theta_star=0.50, sigma=0.30)
        assert results.shape == (3,)
        assert results[0] > results[1] > results[2]

    def test_scalar_input(self):
        result = attack_mass(0.5)
        assert isinstance(float(result), float)


# ── logistic ──────────────────────────────────────────────────────────────────


class TestLogistic:
    """Tests for logistic(x, b0, b1) = 1 / (1 + exp(b0 + b1*x))."""

    def test_midpoint_at_zero_intercept(self):
        # b0=0, b1=anything: logistic(0) = 1/(1+exp(0)) = 0.5
        assert logistic(0, 0, 1) == pytest.approx(0.5, abs=1e-10)
        assert logistic(0, 0, -5) == pytest.approx(0.5, abs=1e-10)
        assert logistic(0, 0, 100) == pytest.approx(0.5, abs=1e-10)

    def test_midpoint_at_negative_b0_over_b1(self):
        # logistic(x, b0, b1) = 0.5 when b0 + b1*x = 0, i.e., x = -b0/b1
        # b0=2, b1=-4 → midpoint at x=0.5
        assert logistic(0.5, 2.0, -4.0) == pytest.approx(0.5, abs=1e-10)

    def test_positive_b1_gives_decreasing_function(self):
        # exp(b0 + b1*x) grows with x when b1>0, so logistic decreases
        x_vals = np.linspace(-2, 2, 20)
        y_vals = logistic(x_vals, 0.0, 2.0)
        assert np.all(np.diff(y_vals) < 0)

    def test_negative_b1_gives_increasing_function(self):
        # When b1<0, logistic is increasing (join rate rises with theta)
        x_vals = np.linspace(-2, 2, 20)
        y_vals = logistic(x_vals, 0.0, -2.0)
        assert np.all(np.diff(y_vals) > 0)

    def test_saturates_to_zero_large_positive_argument(self):
        # b0 + b1*x >> 0 → logistic → 0
        assert logistic(100, 0, 1) == pytest.approx(0.0, abs=1e-10)

    def test_saturates_to_one_large_negative_argument(self):
        # b0 + b1*x << 0 → logistic → 1
        assert logistic(-100, 0, 1) == pytest.approx(1.0, abs=1e-10)

    def test_output_bounded(self):
        x_vals = np.linspace(-10, 10, 100)
        y_vals = logistic(x_vals, 1.0, -3.0)
        assert np.all(y_vals >= 0.0)
        assert np.all(y_vals <= 1.0)

    def test_array_and_scalar_consistent(self):
        x_arr = np.array([0.0, 0.5, 1.0])
        for x in x_arr:
            assert logistic(x, 1.0, -2.0) == pytest.approx(
                logistic(np.array([x]), 1.0, -2.0)[0], abs=1e-12
            )


# ── fit_logistic ───────────────────────────────────────────────────────────────


class TestFitLogistic:
    """Tests for fit_logistic(df, theta_col, jcol).

    Verifies parameter recovery from synthetic data.
    """

    TRUE_B0 = 1.0
    TRUE_B1 = -2.0

    def _synthetic_df(self, noise_sd=0.02, n=50, seed=42):
        rng = np.random.default_rng(seed)
        x = np.linspace(0.0, 1.0, n)
        y_true = logistic(x, self.TRUE_B0, self.TRUE_B1)
        y = np.clip(y_true + rng.normal(0, noise_sd, n), 0.0, 1.0)
        return pd.DataFrame({"theta": x, "join_fraction_valid": y})

    def test_recovers_midpoint(self):
        # Fitted midpoint -b0/b1 should match the true one (0.5) closely
        df = self._synthetic_df()
        popt, _ = fit_logistic(df)
        fitted_midpoint = -popt[0] / popt[1]
        assert fitted_midpoint == pytest.approx(0.5, abs=0.05)

    def test_recovers_parameters_approximately(self):
        df = self._synthetic_df()
        popt, _ = fit_logistic(df)
        assert popt[0] == pytest.approx(self.TRUE_B0, abs=0.15)
        assert popt[1] == pytest.approx(self.TRUE_B1, abs=0.15)

    def test_returns_arrays_of_correct_shape(self):
        df = self._synthetic_df()
        popt, pcov = fit_logistic(df)
        assert popt.shape == (2,)
        assert pcov.shape == (2, 2)

    def test_fallback_on_bad_data(self):
        # All-NaN data: should return zeros rather than raising
        df = pd.DataFrame({"theta": [np.nan] * 5, "join_fraction": [np.nan] * 5})
        popt, pcov = fit_logistic(df)
        assert np.array_equal(popt, [0.0, 0.0])
        assert np.array_equal(pcov, np.zeros((2, 2)))

    def test_uses_join_fraction_valid_by_default(self):
        # When both columns present, join_fraction_valid is preferred
        df = self._synthetic_df()
        df["join_fraction"] = 1 - df["join_fraction_valid"]   # opposite signal
        popt_default, _ = fit_logistic(df)
        popt_explicit, _ = fit_logistic(df, jcol="join_fraction_valid")
        assert np.allclose(popt_default, popt_explicit, atol=1e-6)

    def test_explicit_jcol_overrides_column_selection(self):
        df = self._synthetic_df()
        # Corrupt join_fraction_valid so it has no signal; put signal in other_col
        df["other_col"] = logistic(df["theta"].values, self.TRUE_B0, self.TRUE_B1)
        df["join_fraction_valid"] = np.nan
        popt, _ = fit_logistic(df, jcol="other_col")
        fitted_midpoint = -popt[0] / popt[1]
        assert fitted_midpoint == pytest.approx(0.5, abs=0.05)

    def test_explicit_theta_col(self):
        df = self._synthetic_df()
        df = df.rename(columns={"theta": "my_theta"})
        popt, _ = fit_logistic(df, theta_col="my_theta")
        fitted_midpoint = -popt[0] / popt[1]
        assert fitted_midpoint == pytest.approx(0.5, abs=0.05)


# ── join_col ──────────────────────────────────────────────────────────────────


class TestJoinCol:
    """Tests for join_col(df): column selector helper."""

    def test_prefers_join_fraction_valid_when_present_and_nonempty(self):
        df = pd.DataFrame({
            "join_fraction_valid": [0.5, 0.3, 0.7],
            "join_fraction": [0.4, 0.2, 0.6],
        })
        assert join_col(df) == "join_fraction_valid"

    def test_falls_back_when_valid_col_all_nan(self):
        df = pd.DataFrame({
            "join_fraction_valid": [np.nan, np.nan, np.nan],
            "join_fraction": [0.4, 0.2, 0.6],
        })
        assert join_col(df) == "join_fraction"

    def test_falls_back_when_valid_col_absent(self):
        df = pd.DataFrame({"join_fraction": [0.4, 0.2, 0.6]})
        assert join_col(df) == "join_fraction"

    def test_partial_nan_still_uses_valid_col(self):
        # Even one non-NaN value in join_fraction_valid is enough to prefer it
        df = pd.DataFrame({
            "join_fraction_valid": [np.nan, 0.5, np.nan],
            "join_fraction": [0.4, 0.2, 0.6],
        })
        assert join_col(df) == "join_fraction_valid"

    def test_empty_dataframe_without_valid_col(self):
        df = pd.DataFrame({"join_fraction": pd.Series([], dtype=float)})
        assert join_col(df) == "join_fraction"

    def test_empty_join_fraction_valid_col(self):
        df = pd.DataFrame({
            "join_fraction_valid": pd.Series([], dtype=float),
            "join_fraction": pd.Series([], dtype=float),
        })
        # Empty series has no non-NaN values → falls back
        assert join_col(df) == "join_fraction"
