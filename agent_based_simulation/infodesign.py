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
    clarity_target: Optional[float] = None

    # Additive slider offsets for orthogonalization tests.
    # Negative direction_offset => weaker-state framing; positive => stronger-state framing.
    # Negative coordination_offset => more open/action-oriented tone; positive => quieter/cautious tone.
    direction_offset: float = 0.0
    coordination_offset: float = 0.0

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

    # Domain removal: completely remove specified domains from generated briefings
    # (e.g., (3, 5) to ablate coordination-relevant domains)
    disabled_domains: tuple[int, ...] = ()


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

# ── Common-knowledge vs private × coordination 2×2 ──────────────
# Tests whether common-knowledge framing interacts with coordination cues.

CK_HIGH_COORD = InfoDesignConfig(
    name="ck_high_coord",
    coordination_slope_factor=2.0,
    bandwidth=0.15,
    source_header=(
        "This briefing has been distributed to all citizens. "
        "Everyone has access to the same information."
    ),
)

CK_LOW_COORD = InfoDesignConfig(
    name="ck_low_coord",
    coordination_slope_factor=0.3,
    bandwidth=0.15,
    source_header=(
        "This briefing has been distributed to all citizens. "
        "Everyone has access to the same information."
    ),
)

PRIV_HIGH_COORD = InfoDesignConfig(
    name="priv_high_coord",
    coordination_slope_factor=2.0,
    bandwidth=0.15,
    source_header=(
        "This briefing is based on your personal contacts "
        "and private intelligence."
    ),
)

PRIV_LOW_COORD = InfoDesignConfig(
    name="priv_low_coord",
    coordination_slope_factor=0.3,
    bandwidth=0.15,
    source_header=(
        "This briefing is based on your personal contacts "
        "and private intelligence."
    ),
)

# ── Hard scramble ────────────────────────────────────────────────
# All briefings generated from a single fixed theta (e.g., 0.5),
# ensuring zero correlation between theta and briefing content.

HARD_SCRAMBLE = InfoDesignConfig(
    name="hard_scramble",
    bandwidth=0.15,
)

# ── Common-knowledge framing (valence-held-constant) ───────────────
# Holds the underlying private briefing constant and varies only whether
# the information is framed as public/common knowledge vs private.

CK_PUBLIC = InfoDesignConfig(
    name="ck_public",
    source_header=(
        "NOTE: The key points below have been broadcast nationally and are widely known. "
        "Most citizens have seen the same report and know that others have seen it."
    ),
)

CK_PRIVATE = InfoDesignConfig(
    name="ck_private",
    source_header=(
        "NOTE: The key points below are circulating privately among small circles. "
        "Most citizens likely have not seen this report, and you should not assume others know the same details."
    ),
)

# ── B/C comparative statics (cost/benefit narrative) ─────────────
# Vary perceived cost of failure vs benefit of success in the narrative.
# Theory predicts: higher cost → lower cutoff (less joining).

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

# ── Held-grid narrative payoff conditions ─────────────────────────
# Five levels between high and low cost, all on the same θ-grid (B=C=1).
# Tests whether narrative framing of cost shifts behavior with identical briefings.

BC_VERY_HIGH_COST = InfoDesignConfig(
    name="bc_very_high_cost",
    source_header=(
        "CONTEXT: The regime maintains an extensive network of informants and has "
        "a documented history of disappearances following failed uprisings. Participants' "
        "families are routinely targeted. The cost of unsuccessful action is existential."
    ),
)

BC_MODERATE_HIGH_COST = InfoDesignConfig(
    name="bc_moderate_high_cost",
    source_header=(
        "CONTEXT: Participants in past failed uprisings have faced prison sentences "
        "of several years and loss of employment. The personal cost of unsuccessful "
        "action is significant but not life-threatening."
    ),
)

BC_NEUTRAL = InfoDesignConfig(
    name="bc_neutral",
    source_header=(
        "CONTEXT: The consequences of participating in a failed uprising are uncertain "
        "and vary case by case. Some participants have faced penalties while others "
        "have not. The personal stakes are ambiguous."
    ),
)

BC_MODERATE_LOW_COST = InfoDesignConfig(
    name="bc_moderate_low_cost",
    source_header=(
        "CONTEXT: Recent legal reforms have limited the regime's ability to punish "
        "dissent. Past participants faced only brief questioning and were released "
        "without charges. The personal risk of action is modest."
    ),
)

BC_VERY_LOW_COST = InfoDesignConfig(
    name="bc_very_low_cost",
    source_header=(
        "CONTEXT: The international community has guaranteed safe passage and asylum "
        "for all participants regardless of outcome. Even failed uprisings carry "
        "essentially zero personal cost."
    ),
)

# ── B/C paraphrase variants (robustness check) ───────────────────

BC_HIGH_COST_V2 = InfoDesignConfig(
    name="bc_high_cost_v2",
    source_header=(
        "CONTEXT: In this country, authorities have consistently punished failed "
        "challengers with lengthy imprisonment and confiscation of property. "
        "Family members of participants also face persecution. Acting and failing "
        "carries devastating consequences."
    ),
)

BC_LOW_COST_V2 = InfoDesignConfig(
    name="bc_low_cost_v2",
    source_header=(
        "CONTEXT: A robust network of human rights organizations provides legal "
        "protection to political participants. Historical precedent shows that "
        "even unsuccessful challengers face no lasting repercussions."
    ),
)

# ── Placebo header (irrelevant content control) ──────────────────

BC_PLACEBO = InfoDesignConfig(
    name="bc_placebo",
    source_header=(
        "CONTEXT: The country has experienced significant weather disruptions this year, "
        "including severe flooding in the eastern provinces and an extended drought in "
        "agricultural regions. Infrastructure damage has been substantial."
    ),
)

# ── Publicness × authority 2×2 ───────────────────────────────────
# Crosses {public/private framing} × {authority/anonymous source}.

PUBLIC_AUTHORITY = InfoDesignConfig(
    name="public_authority",
    source_header=(
        "OFFICIAL GOVERNMENT BULLETIN: This assessment has been prepared by the "
        "Ministry of Internal Affairs and distributed to all citizens as a matter "
        "of public record."
    ),
    inject_public_signal=False,
)

PUBLIC_ANONYMOUS = InfoDesignConfig(
    name="public_anonymous",
    source_header=(
        "NOTE: This document has been distributed to all citizens through an "
        "anonymous channel. Every citizen in your area has received an identical copy. "
        "The source of this information is unknown."
    ),
)

PRIVATE_AUTHORITY = InfoDesignConfig(
    name="private_authority",
    source_header=(
        "OFFICIAL GOVERNMENT BULLETIN: This assessment has been prepared by the "
        "Ministry of Internal Affairs and provided to you individually as a "
        "classified briefing. No other citizens have seen this document."
    ),
)

PRIVATE_ANONYMOUS = InfoDesignConfig(
    name="private_anonymous",
    source_header=(
        "NOTE: This document was obtained through an anonymous source and provided "
        "to you alone. No other citizens have seen this information. The source "
        "is unknown."
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

# ── Domain removal ablation designs ──────────────────────────────
# Completely removes specified domains from briefings (vs scramble which
# decorrelates them). Tests whether coordination cues vs state cues drive behavior.
# Domain indices: 0=elite_cohesion, 1=security_forces, 2=money_and_logistics,
#   3=street_mood, 4=information_control, 5=personal_observations,
#   6=diplomatic_signals, 7=institutional_functioning
# The dissent floor (dissent_floor param) affects all remaining domains uniformly.

ABLATE_COORDINATION = InfoDesignConfig(
    name="ablate_coordination",
    disabled_domains=(3, 5),  # street_mood, personal_observations
)

ABLATE_STATE = InfoDesignConfig(
    name="ablate_state",
    disabled_domains=(0, 1, 4, 7),  # elite, security, info_control, institutional
)

# ── Orthogonalized state-evidence × coordination-tone designs ───────
# Hold clarity fixed and shift the factual-state slider independently from the
# coordination-tone slider. These are constant across the θ-grid (large bandwidth)
# so the manipulation is within-briefing rather than concentrated near θ*.

ORTH_WEAK_OPEN = InfoDesignConfig(
    name="orth_weak_open",
    clarity_target=0.85,
    direction_offset=-2.0,
    coordination_offset=-2.0,
    bandwidth=10.0,
)

ORTH_WEAK_QUIET = InfoDesignConfig(
    name="orth_weak_quiet",
    clarity_target=0.85,
    direction_offset=-2.0,
    coordination_offset=2.0,
    bandwidth=10.0,
)

ORTH_STRONG_OPEN = InfoDesignConfig(
    name="orth_strong_open",
    clarity_target=0.85,
    direction_offset=2.0,
    coordination_offset=-2.0,
    bandwidth=10.0,
)

ORTH_STRONG_QUIET = InfoDesignConfig(
    name="orth_strong_quiet",
    clarity_target=0.85,
    direction_offset=2.0,
    coordination_offset=2.0,
    bandwidth=10.0,
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
    "ck_public": CK_PUBLIC,
    "ck_private": CK_PRIVATE,
    "bc_high_cost": BC_HIGH_COST,
    "bc_low_cost": BC_LOW_COST,
    "bc_very_high_cost": BC_VERY_HIGH_COST,
    "bc_moderate_high_cost": BC_MODERATE_HIGH_COST,
    "bc_neutral": BC_NEUTRAL,
    "bc_moderate_low_cost": BC_MODERATE_LOW_COST,
    "bc_very_low_cost": BC_VERY_LOW_COST,
    "bc_high_cost_v2": BC_HIGH_COST_V2,
    "bc_low_cost_v2": BC_LOW_COST_V2,
    "bc_placebo": BC_PLACEBO,
    "public_authority": PUBLIC_AUTHORITY,
    "public_anonymous": PUBLIC_ANONYMOUS,
    "private_authority": PRIVATE_AUTHORITY,
    "private_anonymous": PRIVATE_ANONYMOUS,
    "censor_upper_known": UPPER_CENSORSHIP_KNOWN,
    "within_scramble": WITHIN_SCRAMBLE,
    "domain_scramble_coord": DOMAIN_SCRAMBLE_COORDINATION,
    "domain_scramble_state": DOMAIN_SCRAMBLE_STATE,
    "ablate_coordination": ABLATE_COORDINATION,
    "ablate_state": ABLATE_STATE,
    "orth_weak_open": ORTH_WEAK_OPEN,
    "orth_weak_quiet": ORTH_WEAK_QUIET,
    "orth_strong_open": ORTH_STRONG_OPEN,
    "orth_strong_quiet": ORTH_STRONG_QUIET,
    "ck_high_coord": CK_HIGH_COORD,
    "ck_low_coord": CK_LOW_COORD,
    "priv_high_coord": PRIV_HIGH_COORD,
    "priv_low_coord": PRIV_LOW_COORD,
    "hard_scramble": HARD_SCRAMBLE,
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
        if self.config.clarity_target is not None:
            params["clarity_override"] = self.config.clarity_target
        params["direction_offset"] = params.get("direction_offset", 0.0) + w * self.config.direction_offset
        params["coordination_offset"] = params.get("coordination_offset", 0.0) + w * self.config.coordination_offset

        # Merge disabled domains from config with any from base_params
        if self.config.disabled_domains:
            existing = set(params.get("disabled_domains", ()))
            params["disabled_domains"] = tuple(existing | set(self.config.disabled_domains))

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
        "direction_offset": calibrated.get("direction_offset", 0.0),
        "coordination_offset": calibrated.get("coordination_offset", 0.0),
        "clarity_override": calibrated.get("clarity_override", None),
        "seed": seed,
    }
    return params
