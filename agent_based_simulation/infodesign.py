"""
Information design module for Paper 2: Bayesian persuasion in LLM global games.

A social planner controls the *information structure* — how states map to signals —
but not agents' actions. Three designs modulate BriefingGenerator params as a
function of θ to shift coordination outcomes:

  1. Stability-maximizing — increase noise near θ*, suppress coordination
  2. Instability-maximizing — sharpen signals near θ*, amplify coordination
  3. Public signal injection — shared "news bulletin" appended to private briefings

All designs use a Gaussian proximity weight to concentrate manipulation near θ*:
    w(θ) = exp(-((θ - θ*) / bandwidth)²)

Theory reference: Bergemann & Morris (2019), "Information Design: A Unified
Perspective," Journal of Economic Literature.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .briefing import BriefingGenerator, Briefing


# ── Proximity weighting ──────────────────────────────────────────────

def _proximity_weight(theta: float, theta_star: float, bandwidth: float) -> float:
    """Gaussian weight ∈ [0, 1], peaks at θ = θ*.

    w(θ) = exp(-((θ - θ*) / bandwidth)²)
    """
    return float(np.exp(-((theta - theta_star) / bandwidth) ** 2))


# ── Design configuration ─────────────────────────────────────────────

@dataclass
class InfoDesignConfig:
    """Specification of an information design intervention.

    Each field ending in ``_factor`` is a multiplicative modifier applied
    to the base BriefingGenerator param at full proximity weight.
    Fields ending in ``_target`` are absolute values blended toward.

    At distance from θ*, params smoothly revert to baseline.
    """
    name: str

    # Multiplicative modifiers (applied at w=1, lerped toward 1.0 at w=0)
    clarity_width_factor: float = 1.0
    direction_slope_factor: float = 1.0
    coordination_slope_factor: float = 1.0

    # Absolute targets (blended: base*(1-w) + target*w)
    dissent_floor_target: Optional[float] = None

    # Public signal injection
    inject_public_signal: bool = False
    public_signal_n_observations: int = 4

    # Censorship: binary signal pooling (Kolotilin et al. 2022)
    #   None  = no censorship
    #   "upper" = censor when θ ≤ θ* (stability: hide weakness near tipping point)
    #   "lower" = censor when θ ≥ θ* (instability: hide strength near tipping point)
    censorship_mode: Optional[str] = None

    # Provenance: source attribution header prepended to briefings
    source_header: Optional[str] = None

    # Rhetoric: shift phrase ladder rungs (positive = more intense/urgent)
    rhetoric_bias: float = 0.0

    # Proximity bandwidth
    bandwidth: float = 0.15

    # Within-briefing scramble: shuffle observation bullets within each briefing
    shuffle_observations: bool = False

    # Domain-group scramble: indices of observation domains to scramble across agents
    # (e.g., (3, 5) for coordination-relevant domains street_mood + personal_observations)
    scramble_domain_indices: Optional[tuple[int, ...]] = None


# ── Pre-built designs ────────────────────────────────────────────────

STABILITY_DESIGN = InfoDesignConfig(
    name="stability",
    clarity_width_factor=4.0,       # 4× wider ambiguous zone → noisier signals
    direction_slope_factor=0.25,    # flatter direction → weaker signal gradient
    dissent_floor_target=0.45,      # raise dissent → more contrary evidence
    bandwidth=0.15,
)

INSTABILITY_DESIGN = InfoDesignConfig(
    name="instability",
    clarity_width_factor=0.15,      # narrow ambiguous zone → crisper signals
    direction_slope_factor=3.0,     # steeper direction → sharper signal gradient
    dissent_floor_target=0.05,      # lower dissent → less contrary evidence
    bandwidth=0.15,
)

PUBLIC_SIGNAL_DESIGN = InfoDesignConfig(
    name="public_signal",
    inject_public_signal=True,
    public_signal_n_observations=4,
    bandwidth=0.15,
)

# ── Single-channel decomposition designs ─────────────────────────────
# Each isolates ONE manipulation channel from the bundled stability design.

STABILITY_CLARITY_ONLY = InfoDesignConfig(
    name="stability_clarity",
    clarity_width_factor=4.0,
    bandwidth=0.15,
)

STABILITY_DIRECTION_ONLY = InfoDesignConfig(
    name="stability_direction",
    direction_slope_factor=0.25,
    bandwidth=0.15,
)

STABILITY_DISSENT_ONLY = InfoDesignConfig(
    name="stability_dissent",
    dissent_floor_target=0.45,
    bandwidth=0.15,
)

# ── Censorship designs (Kolotilin et al. 2022) ──────────────────────
# Binary: censor (pool to neutral z=0) on one side of θ*, reveal on other.

UPPER_CENSORSHIP_DESIGN = InfoDesignConfig(
    name="censor_upper",
    censorship_mode="upper",
    bandwidth=0.15,
)

LOWER_CENSORSHIP_DESIGN = InfoDesignConfig(
    name="censor_lower",
    censorship_mode="lower",
    bandwidth=0.15,
)

# ── Provenance designs (source attribution treatment) ──────────────

PROVENANCE_INDEPENDENT = InfoDesignConfig(
    name="provenance_independent",
    source_header="SOURCE: Independent international observers (multi-source corroboration)",
)

PROVENANCE_STATE = InfoDesignConfig(
    name="provenance_state",
    source_header="SOURCE: Ministry of Information, Government of Silvaria",
)

PROVENANCE_SOCIAL = InfoDesignConfig(
    name="provenance_social",
    source_header="SOURCE: Aggregated social media posts and citizen reports",
)

# ── Rhetoric designs (hot/cold language intensity) ─────────────────

RHETORIC_HOT = InfoDesignConfig(
    name="rhetoric_hot",
    rhetoric_bias=2.0,
)

RHETORIC_COLD = InfoDesignConfig(
    name="rhetoric_cold",
    rhetoric_bias=-2.0,
)

# ── Coordination designs (counterbalanced) ───────────────────────
# Vary coordination_slope to amplify/suppress coordination cues.

COORDINATION_AMPLIFIED = InfoDesignConfig(
    name="coord_amplified",
    coordination_slope_factor=2.0,
    bandwidth=0.15,
)

COORDINATION_SUPPRESSED = InfoDesignConfig(
    name="coord_suppressed",
    coordination_slope_factor=0.3,
    bandwidth=0.15,
)

# ── B/C comparative statics (cost/benefit narrative) ─────────────
# Vary perceived cost of failure vs benefit of success in the narrative.
# Theory predicts: higher cost → higher cutoff (less joining).

BC_HIGH_COST = InfoDesignConfig(
    name="bc_high_cost",
    source_header=(
        "CONTEXT: Failed uprisings in this country have historically resulted "
        "in severe reprisals---imprisonment, asset seizure, and retaliation "
        "against families. The personal cost of unsuccessful action is "
        "extremely high."
    ),
)

BC_LOW_COST = InfoDesignConfig(
    name="bc_low_cost",
    source_header=(
        "CONTEXT: International observers are monitoring the situation closely. "
        "Even in failed uprisings, participants have historically faced minimal "
        "consequences---brief detentions at most. The personal risk of action "
        "is low."
    ),
)

# ── Censorship with common knowledge ──────────────────────────────
# Agents know censorship is occurring, enabling Bayesian updating about
# the censorship rule rather than just observing bland text.

UPPER_CENSORSHIP_KNOWN = InfoDesignConfig(
    name="censor_upper_known",
    censorship_mode="upper",
    bandwidth=0.15,
    source_header=(
        "NOTE: Independent analysts report that regime censors are suppressing "
        "unfavorable intelligence above a certain severity threshold. The "
        "information below may be filtered."
    ),
)

# ── Within-briefing falsification ─────────────────────────────────
# Tests whether bullet ordering or domain-specific content drives correlation.

WITHIN_SCRAMBLE = InfoDesignConfig(
    name="within_scramble",
    shuffle_observations=True,
)

DOMAIN_SCRAMBLE_COORDINATION = InfoDesignConfig(
    name="domain_scramble_coord",
    scramble_domain_indices=(3, 5),  # street_mood, personal_observations
)

DOMAIN_SCRAMBLE_STATE = InfoDesignConfig(
    name="domain_scramble_state",
    scramble_domain_indices=(0, 1, 4, 7),  # elite, security, info_control, institutional
)

# Falsification designs: scramble and flip within each info design
# These reuse the same config but are run with signal_mode="scramble"/"flip"
# in the experiment runner.

ALL_DESIGNS = {
    "baseline": None,  # no manipulation
    "stability": STABILITY_DESIGN,
    "instability": INSTABILITY_DESIGN,
    "public_signal": PUBLIC_SIGNAL_DESIGN,
    "stability_clarity": STABILITY_CLARITY_ONLY,
    "stability_direction": STABILITY_DIRECTION_ONLY,
    "stability_dissent": STABILITY_DISSENT_ONLY,
    "censor_upper": UPPER_CENSORSHIP_DESIGN,
    "censor_lower": LOWER_CENSORSHIP_DESIGN,
    "provenance_independent": PROVENANCE_INDEPENDENT,
    "provenance_state": PROVENANCE_STATE,
    "provenance_social": PROVENANCE_SOCIAL,
    "rhetoric_hot": RHETORIC_HOT,
    "rhetoric_cold": RHETORIC_COLD,
    "coord_amplified": COORDINATION_AMPLIFIED,
    "coord_suppressed": COORDINATION_SUPPRESSED,
    "bc_high_cost": BC_HIGH_COST,
    "bc_low_cost": BC_LOW_COST,
    "censor_upper_known": UPPER_CENSORSHIP_KNOWN,
    "within_scramble": WITHIN_SCRAMBLE,
    "domain_scramble_coord": DOMAIN_SCRAMBLE_COORDINATION,
    "domain_scramble_state": DOMAIN_SCRAMBLE_STATE,
}


# ── Theta-adaptive briefing generator ────────────────────────────────

class ThetaAdaptiveBriefingGenerator:
    """Wraps BriefingGenerator with θ-dependent parameter modulation.

    Usage in experiment loop:
        gen = ThetaAdaptiveBriefingGenerator(base_params, config, theta_star)
        gen.set_theta(theta)  # call once per period
        briefing = gen.generate(z_score, agent_id, period)  # same signature
    """

    def __init__(
        self,
        base_params: dict,
        config: InfoDesignConfig,
        theta_star: float,
    ):
        self.base_params = dict(base_params)  # copy to avoid mutation
        self.config = config
        self.theta_star = theta_star
        self._current_gen: Optional[BriefingGenerator] = None
        self._current_theta: Optional[float] = None

    def set_theta(self, theta: float) -> None:
        """Modulate params based on proximity of θ to θ* and rebuild generator."""
        self._current_theta = theta
        w = _proximity_weight(theta, self.theta_star, self.config.bandwidth)

        params = dict(self.base_params)

        # Multiplicative modifiers: param = base * lerp(1.0, factor, w)
        params["clarity_width"] = max(1e-8, params.get("clarity_width", 1.0) * (
            1.0 + w * (self.config.clarity_width_factor - 1.0)
        ))
        params["direction_slope"] = max(1e-8, params.get("direction_slope", 0.8) * (
            1.0 + w * (self.config.direction_slope_factor - 1.0)
        ))
        params["coordination_slope"] = max(1e-8, params.get("coordination_slope", 0.6) * (
            1.0 + w * (self.config.coordination_slope_factor - 1.0)
        ))

        # Absolute target blending: param = base*(1-w) + target*w
        if self.config.dissent_floor_target is not None:
            base_dissent = params.get("dissent_floor", 0.25)
            params["dissent_floor"] = (
                base_dissent * (1.0 - w) + self.config.dissent_floor_target * w
            )

        self._current_gen = BriefingGenerator(**params)

    def generate(self, z_score: float, agent_id: int = 0, period: int = 0) -> Briefing:
        """Generate briefing — same signature as BriefingGenerator.generate().

        Censorship (if active) replaces z_score with 0.0 when θ falls in the
        pooled region, producing a neutral briefing that hides state information.
        """
        if self._current_gen is None:
            raise RuntimeError(
                "Must call set_theta() before generate(). "
                "The planner sets θ once per period before the signal assignment loop."
            )

        # Binary censorship: pool z-score to 0 on the censored side of θ*
        cmode = self.config.censorship_mode
        if cmode is not None and self._current_theta is not None:
            if cmode == "upper" and self._current_theta <= self.theta_star:
                z_score = 0.0
            elif cmode == "lower" and self._current_theta >= self.theta_star:
                z_score = 0.0

        return self._current_gen.generate(z_score, agent_id, period)


# ── Public signal generator ──────────────────────────────────────────

class PublicSignal:
    """Generate a shared "news bulletin" from θ.

    The public signal is a briefing generated at z_score = (θ - z) / σ
    (i.e., the true state's z-score) with reduced observations, providing
    a common-knowledge channel to all agents.
    """

    def __init__(self, base_params: dict, n_observations: int = 4, seed: int = 42):
        params = dict(base_params)
        params["n_observations"] = n_observations
        params["seed"] = seed
        self._gen = BriefingGenerator(**params)

    def generate(self, theta: float, z: float, sigma: float, period: int = 0, bulletin_seed: int = 9999) -> str:
        """Generate a public news bulletin text from the true state.

        Returns rendered briefing text suitable for appending to private briefings.
        """
        z_score = (theta - z) / sigma
        briefing = self._gen.generate(z_score, agent_id=bulletin_seed, period=period)
        rendered = briefing.render()
        return (
            "\n\n--- PUBLIC NEWS BULLETIN ---\n"
            "The following report has been broadcast on state media and is available "
            "to all citizens:\n\n"
            f"{rendered}\n"
            "--- END BULLETIN ---"
        )


# ── Helper: extract base params dict from calibrated params ──────────

def base_params_from_calibrated(calibrated: dict, seed: int = None) -> dict:
    """Extract BriefingGenerator kwargs from a calibrated params dict."""
    params = {
        "cutoff_center": calibrated.get("cutoff_center", 0.0),
        "clarity_width": calibrated.get("clarity_width", 1.0),
        "direction_slope": calibrated.get("direction_slope", 0.8),
        "coordination_slope": calibrated.get("coordination_slope", 0.6),
        "dissent_floor": calibrated.get("dissent_floor", 0.25),
        "mixed_cue_clarity": calibrated.get("mixed_cue_clarity", 0.5),
        "bottomline_cuts": calibrated.get("bottomline_cuts"),
        "unclear_cuts": calibrated.get("unclear_cuts"),
        "coordination_cuts": calibrated.get("coordination_cuts"),
        "coordination_blend_prob": calibrated.get("coordination_blend_prob", 0.6),
        "language_variant": calibrated.get("language_variant", "baseline"),
        "seed": seed,
    }
    return params
