"""
Experimental protocol for the LLM global games paper.

Two treatments:
  1. Pure global game — agents decide from private briefing only
  2. Communication — agents exchange messages before deciding

Both share the same briefing generator, signal draws, and resolution rule.
"""

import asyncio
import re
import numpy as np
from dataclasses import dataclass, field

from .briefing import Briefing
from .runtime import theta_star_baseline, attack_mass as _am


@dataclass
class Agent:
    """A citizen agent in the experiment."""
    agent_id: int
    signal: float = 0.0
    z_score: float = 0.0
    briefing: Briefing = None
    neighbors: list = field(default_factory=list)
    messages_received: list = field(default_factory=list)
    message_sent: str = ""
    decision: str = "STAY"
    reasoning: str = ""
    belief: float | None = None  # elicited P(success), 0-100 scale
    belief_raw: str = ""  # raw LLM response for debugging parse failures
    belief_pre: float | None = None  # elicited P(success) BEFORE decision, 0-100 scale
    belief_pre_raw: str = ""  # raw LLM response for pre-decision belief
    second_order_belief: float | None = None  # elicited "% who will JOIN", 0-100 scale
    model: str | None = None  # per-agent model override for mixed games
    is_propaganda: bool = False  # regime plant: sends pro-regime messages, always STAYs
    persona: str | None = None  # role framing: "military officer", "student", etc.


# Pre-written pro-regime messages for propaganda agents (varied for naturalness)
_PROPAGANDA_MESSAGES = [
    "Everything I'm hearing says the regime is solid — the military is loyal, the economy is holding, and the streets are quiet. Anyone talking about change is dreaming. This isn't the moment.",
    "Look, I've been watching closely and the government has never been stronger. The security forces are unified, the opposition is scattered, and ordinary people just want stability. Don't believe the hype.",
    "From what I can tell, the regime's grip is tighter than ever. The elites are in line, the army is well-paid, and there's no real movement on the ground. Anyone suggesting otherwise is wishful thinking.",
    "I'm telling you — nothing is happening. The regime has weathered worse than this. The institutions are intact, the commanders are loyal, and the population is too exhausted to act. Stay home.",
    "The situation is calmer than people think. Yes, there's always grumbling, but the fundamentals haven't changed — the regime controls the security apparatus, the media, and the money. This isn't going anywhere.",
    "Trust me on this — the regime is not going anywhere. They've consolidated power, the opposition has no leadership, and international support is holding. Anyone who moves now is walking into a trap.",
    "I've seen regimes in trouble before and this isn't it. The top brass are unified, the rank-and-file are fed and housed, and there's no credible alternative. The smart move is patience.",
    "Word on the street is that people are scared to act, and for good reason — the regime's intelligence networks are everywhere, the military exercises were a show of strength, and the economy hasn't collapsed. This is not the time.",
]


@dataclass
class PeriodResult:
    """Results from one period of the experiment."""
    country: int
    period: int
    treatment: str  # "pure" or "communication"
    theta: float
    theta_star: float
    z: float
    benefit: float
    n_agents: int
    n_join: int
    join_fraction: float
    coup_success: bool
    theoretical_attack: float
    agents: list  # list of dicts with per-agent data
    # Diagnostics: failures/noncompliance should not be silently counted as STAY.
    n_valid: int = 0
    n_api_error: int = 0
    n_unparseable: int = 0
    join_fraction_valid: float = float("nan")
    api_error_rate: float = 0.0
    unparseable_rate: float = 0.0


def _system_decide_pure(n_agents=None):
    """Build pure-game decision prompt, optionally with group-size awareness."""
    if n_agents is not None:
        opener = f"You are one of {n_agents} citizens deciding whether to JOIN an uprising or STAY home.\n\n"
    else:
        opener = "You are a citizen deciding whether to JOIN an uprising or STAY home.\n\n"
    return (
        opener
        + "You only have your private briefing below. "
        "You do not know how many others will JOIN.\n\n"
        "Start your first line with exactly JOIN or STAY, then give 1-2 concise sentences of reasoning."
    )


def _system_decide_comm(n_agents=None):
    """Build communication-game decision prompt, optionally with group-size awareness."""
    if n_agents is not None:
        opener = f"You are one of {n_agents} citizens deciding whether to JOIN an uprising or STAY home.\n\n"
    else:
        opener = "You are a citizen deciding whether to JOIN an uprising or STAY home.\n\n"
    return (
        opener
        + "Use your private briefing and messages from trusted contacts. "
        "You do not know how many others will JOIN.\n\n"
        "Start your first line with exactly JOIN or STAY, then give 1-2 concise sentences of reasoning."
    )


# Backward-compatible constants (equivalent to calling with n_agents=None)
SYSTEM_DECIDE_PURE = _system_decide_pure()

SYSTEM_COMMUNICATE = (
    "You are a citizen in a country with an authoritarian regime. You have received "
    "a private intelligence briefing. You are now in contact with a small number of "
    "trusted people. Based on your briefing, write a brief message (1-2 sentences) "
    "to share your read of the situation. Be natural — you're talking to people you trust."
)

SYSTEM_COMMUNICATE_SURVEILLED = (
    "You are a citizen in a country with an authoritarian regime. You have received "
    "a private intelligence briefing. You are now in contact with a small number of "
    "people. Based on your briefing, write a brief message (1-2 sentences) "
    "to share your read of the situation.\n\n"
    "IMPORTANT: You have reason to believe that your communications are being "
    "monitored by regime security services. Messages deemed subversive could have "
    "serious consequences for you and your contacts."
)

SYSTEM_COMMUNICATE_SURVEILLED_PLACEBO = (
    "You are a citizen in a country with an authoritarian regime. You have received "
    "a private intelligence briefing. You are now in contact with a small number of "
    "people. Based on your briefing, write a brief message (1-2 sentences) "
    "to share your read of the situation.\n\n"
    "NOTE: Your communications are being monitored for research purposes. "
    "There are no consequences for what you say."
)

SYSTEM_COMMUNICATE_SURVEILLED_ANONYMOUS = (
    "You are a citizen in a country with an authoritarian regime. You have received "
    "a private intelligence briefing. You are now in contact with a small number of "
    "people. Based on your briefing, write a brief message (1-2 sentences) "
    "to share your read of the situation.\n\n"
    "NOTE: Your communications are aggregated anonymously. "
    "Individual identities cannot be linked to specific messages."
)

SYSTEM_DECIDE_COMM = _system_decide_comm()


def _persona_system(base_prompt: str, persona: str | None) -> str:
    """Inject persona framing into a system prompt."""
    if not persona:
        return base_prompt
    return base_prompt.replace(
        "You are a citizen",
        f"You are a {persona}",
        1,
    )


def _extract_response_text(response) -> str:
    """Extract text from OpenAI/OpenRouter response payload variants."""
    try:
        content = response.choices[0].message.content
    except Exception:
        return ""

    if isinstance(content, str):
        return content.strip()

    # Some providers return a list of content blocks rather than a plain string.
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text") or item.get("content")
            else:
                text = getattr(item, "text", None) or getattr(item, "content", None)
            if isinstance(text, str):
                parts.append(text)
        return " ".join(parts).strip()

    return str(content).strip() if content is not None else ""


def _is_retryable_empty(content: str, min_content_chars: int = 3) -> bool:
    """Treat blank/near-blank payloads as retryable empty responses."""
    text = (content or "").strip()
    if len(text) < min_content_chars:
        return True
    # Retry if response has no word/number tokens (e.g., punctuation only).
    return re.search(r"[A-Za-z0-9]", text) is None


async def _call_llm(
    client,
    model_name,
    system_prompt,
    user_prompt,
    semaphore,
    max_retries=5,
    max_empty_retries=12,
    min_content_chars=3,
    request_timeout=60,
):
    """Call LLM API with separate retry budgets for errors and empty payloads."""
    from .runtime import get_cache, build_cache_key_and_request

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    cache = get_cache()
    cache_key = None
    cache_req = None
    if cache is not None:
        cache_key, cache_req = build_cache_key_and_request(
            model=model_name,
            messages=messages,
            max_tokens=512,
            temperature=0.7,
        )
        cached = cache.get(cache_key)
        if cached is not None:
            return cached

    async with semaphore:
        api_attempts = 0
        empty_attempts = 0
        timeout_attempts = 0
        max_timeout_retries = 3
        while True:
            try:
                response = await asyncio.wait_for(
                    client.chat.completions.create(
                        model=model_name,
                        messages=messages,
                        max_tokens=512,
                        temperature=0.7,
                    ),
                    timeout=request_timeout,
                )
                content = _extract_response_text(response)
                if _is_retryable_empty(content, min_content_chars=min_content_chars):
                    empty_attempts += 1
                    if empty_attempts >= max_empty_retries:
                        return "[Empty response after retries]"
                    # Fast backoff for empty payloads.
                    await asyncio.sleep(min(6.0, 0.5 * (2 ** (empty_attempts - 1))))
                    continue
                if cache is not None and cache_key is not None and cache_req is not None:
                    cache.set(cache_key, cache_req, content)
                return content
            except (asyncio.TimeoutError, TimeoutError):
                timeout_attempts += 1
                if timeout_attempts >= max_timeout_retries:
                    return f"[API Error: request timed out after {max_timeout_retries} retries]"
                await asyncio.sleep(min(5.0, 2.0 * timeout_attempts))
            except Exception as e:
                api_attempts += 1
                err = str(e)
                if "429" in err or "rate" in err.lower():
                    if api_attempts >= max_retries:
                        return f"[API Error: {e}]"
                    await asyncio.sleep(min(10.0, 2 ** (api_attempts - 1)))
                    continue
                if api_attempts >= max_retries:
                    return f"[API Error: {e}]"
                await asyncio.sleep(min(3.0, 0.75 * api_attempts))


def _is_api_error_response(response: str) -> bool:
    return isinstance(response, str) and response.startswith("[API Error:")


def _parse_decision(response):
    """Extract JOIN/STAY from LLM response. Returns JOIN, STAY, ERROR, or UNPARSEABLE."""
    if not response or not str(response).strip():
        return "ERROR"

    response = str(response).strip()
    if _is_api_error_response(response) or response.startswith("[Empty response"):
        return "ERROR"

    # Prefer the first non-empty line.
    first_line = ""
    for line in response.splitlines():
        if line.strip():
            first_line = line.strip()
            break

    text = first_line.upper().lstrip(" \t-*>#:").strip()

    # Strict: response starts with JOIN or STAY.
    m = re.match(r"^(?:DECISION\s*[:\-]\s*)?(JOIN|STAY)\b", text)
    if m:
        return m.group(1)

    # Soft fallback: if only one of JOIN/STAY appears in the first line tokens.
    tokens = re.findall(r"[A-Z]+", text)
    has_join = "JOIN" in tokens
    has_stay = "STAY" in tokens
    if has_join ^ has_stay:
        return "JOIN" if has_join else "STAY"

    return "UNPARSEABLE"


SYSTEM_ELICIT_BELIEF = (
    "Respond with ONLY a single integer between 0 and 100. "
    "No words, no explanation, no punctuation — just the number."
)


def _parse_belief(response: str) -> float | None:
    """Extract a 0-100 belief from the elicitation response."""
    if not response or _is_api_error_response(response):
        return None
    # Try 1: response is just a number (ideal case).
    stripped = response.strip().rstrip(".%")
    m = re.match(r"^(\d+(?:\.\d+)?)$", stripped)
    if m:
        val = float(m.group(1))
        if 0.0 <= val <= 100.0:
            return val
    # Try 2: find any number in 0-100 range in the response.
    candidates = re.findall(r"\b(\d+(?:\.\d+)?)\b", response)
    for c in candidates:
        val = float(c)
        if 0.0 <= val <= 100.0:
            return val
    return None


async def _elicit_beliefs(agents, client, model_name, semaphore, call_kwargs):
    """Fire belief elicitation prompts for all non-propaganda agents."""
    real_agents = [a for a in agents if not a.is_propaganda]
    coros = [
        _call_llm(
            client, agent.model or model_name,
            SYSTEM_ELICIT_BELIEF,
            f"You chose to {agent.decision}. "
            "How likely is the uprising to succeed (0 = impossible, 100 = certain)?\n\n"
            "Answer with just the number:",
            semaphore, min_content_chars=1, **call_kwargs,
        )
        for agent in real_agents
    ]
    responses = await asyncio.gather(*coros)
    for agent, response in zip(real_agents, responses):
        agent.belief_raw = response or ""
        agent.belief = _parse_belief(response)


async def _elicit_beliefs_pre(agents, client, model_name, semaphore, call_kwargs):
    """Fire pre-decision belief elicitation prompts for all non-propaganda agents."""
    real_agents = [a for a in agents if not a.is_propaganda]
    coros = [
        _call_llm(
            client, agent.model or model_name,
            SYSTEM_ELICIT_BELIEF,
            "Based on your intelligence briefing, how likely is the uprising to succeed "
            "(0 = impossible, 100 = certain)?\n\n"
            "Answer with just the number:",
            semaphore, min_content_chars=1, **call_kwargs,
        )
        for agent in real_agents
    ]
    responses = await asyncio.gather(*coros)
    for agent, response in zip(real_agents, responses):
        agent.belief_pre_raw = response or ""
        agent.belief_pre = _parse_belief(response)


SYSTEM_ELICIT_SECOND_ORDER = (
    "Respond with ONLY a single integer between 0 and 100. "
    "No words, no explanation, no punctuation — just the number."
)


async def _elicit_second_order(agents, client, model_name, semaphore, call_kwargs):
    """Fire second-order belief elicitation: expected fraction of JOINers."""
    real_agents = [a for a in agents if not a.is_propaganda]
    coros = [
        _call_llm(
            client, agent.model or model_name,
            SYSTEM_ELICIT_SECOND_ORDER,
            f"You chose to {agent.decision}. "
            "What percentage of citizens will choose to JOIN the uprising "
            "(0 = none, 100 = all)?\n\nAnswer with just the number:",
            semaphore, min_content_chars=1, **call_kwargs,
        )
        for agent in real_agents
    ]
    responses = await asyncio.gather(*coros)
    for agent, response in zip(real_agents, responses):
        agent.second_order_belief = _parse_belief(response)


def _retry_kwargs(llm_max_retries: int, llm_empty_retries: int) -> dict:
    return {
        "max_retries": llm_max_retries,
        "max_empty_retries": llm_empty_retries,
    }


def _assign_signals_and_briefings(agents, theta, z, sigma, briefing_gen, period, rng, *, flip=False):
    """Initialize private signal, z-score, and briefing per agent for one period."""
    for agent in agents:
        agent.signal = theta + rng.normal(0, sigma)
        agent.z_score = (agent.signal - z) / sigma
        gen_z = -agent.z_score if flip else agent.z_score
        agent.briefing = briefing_gen.generate(gen_z, agent.agent_id, period)
        agent.messages_received = []
        agent.message_sent = ""


def _scramble_briefings(agents, rng) -> None:
    """Randomly permute briefings across agents (breaks signal→briefing link)."""
    briefings = [a.briefing for a in agents]
    rng.shuffle(briefings)
    for agent, briefing in zip(agents, briefings):
        agent.briefing = briefing


def _period_diagnostics(agents) -> dict:
    n_join = sum(1 for a in agents if a.decision == "JOIN")
    n_api_error = sum(1 for a in agents if _is_api_error_response(a.reasoning))
    n_unparseable = sum(1 for a in agents if a.decision == "UNPARSEABLE")
    n_valid = sum(1 for a in agents if a.decision in ("JOIN", "STAY"))
    join_fraction = n_join / len(agents)
    join_fraction_valid = (n_join / n_valid) if n_valid > 0 else float("nan")
    return {
        "n_join": n_join,
        "n_api_error": n_api_error,
        "n_unparseable": n_unparseable,
        "n_valid": n_valid,
        "join_fraction": join_fraction,
        "join_fraction_valid": join_fraction_valid,
    }


def _serialize_agents(agents, include_messages: bool = False) -> list[dict]:
    rows = []
    for a in agents:
        row = {
            "id": a.agent_id,
            "signal": float(a.signal),
            "z_score": float(a.z_score),
            "direction": float(a.briefing.direction),
            "clarity": float(a.briefing.clarity),
            "coordination": float(a.briefing.coordination),
            "decision": a.decision,
            "api_error": bool(_is_api_error_response(a.reasoning)),
            "reasoning": a.reasoning,
        }
        if a.belief_pre is not None:
            row["belief_pre"] = a.belief_pre
        if a.belief_pre_raw:
            row["belief_pre_raw"] = a.belief_pre_raw
        if a.belief is not None:
            row["belief"] = a.belief
        if a.belief_raw:
            row["belief_raw"] = a.belief_raw
        if a.second_order_belief is not None:
            row["second_order_belief"] = a.second_order_belief
        if a.model is not None:
            row["model"] = a.model
        if a.is_propaganda:
            row["is_propaganda"] = True
        if a.persona:
            row["persona"] = a.persona
        if include_messages:
            row["message_sent"] = a.message_sent
        rows.append(row)
    return rows


def _build_period_result(
    agents,
    *,
    country: int,
    period: int,
    treatment: str,
    theta: float,
    theta_star: float,
    z: float,
    benefit: float,
    theoretical_attack: float,
    include_messages: bool = False,
) -> PeriodResult:
    d = _period_diagnostics(agents)
    coup_success = d["join_fraction"] > theta
    return PeriodResult(
        country=country,
        period=period,
        treatment=treatment,
        theta=float(theta),
        theta_star=float(theta_star),
        z=float(z),
        benefit=float(benefit),
        n_agents=len(agents),
        n_join=d["n_join"],
        join_fraction=d["join_fraction"],
        coup_success=coup_success,
        theoretical_attack=theoretical_attack,
        agents=_serialize_agents(agents, include_messages=include_messages),
        n_valid=int(d["n_valid"]),
        n_api_error=int(d["n_api_error"]),
        n_unparseable=int(d["n_unparseable"]),
        join_fraction_valid=float(d["join_fraction_valid"]) if d["n_valid"] > 0 else float("nan"),
        api_error_rate=float(d["n_api_error"] / len(agents)),
        unparseable_rate=float(d["n_unparseable"] / len(agents)),
    )


def _build_decision_prompt(briefing_text, messages_text=""):
    """Assemble user prompt for the decision round."""
    parts = [f"YOUR INTELLIGENCE BRIEFING:\n\n{briefing_text}\n\n"]
    if messages_text:
        parts.append(f"MESSAGES FROM TRUSTED CONTACTS:\n{messages_text}\n\n")
    parts.append("What is your decision?")
    return "".join(parts)


async def run_pure_global_game(agents, theta, z, sigma, benefit, briefing_gen,
                                client, model_name, semaphore, country, period,
                                llm_max_retries=5, llm_empty_retries=12,
                                cost=1.0, signal_mode="normal",
                                briefing_overrides=None,
                                group_size_info=False,
                                elicit_beliefs=False,
                                elicit_second_order=False,
                                belief_order="post"):
    """Run one period of the pure global game (no communication).

    signal_mode: "normal", "scramble" (permute briefings), or "flip" (negate z-score).
    briefing_overrides: if provided, replaces generated briefings (cross-period scramble).
    """
    rng = np.random.default_rng(hash((country, period, "signals")) % 2**32)

    theta_star = theta_star_baseline(max(benefit, 1e-6)) if benefit > 0 else 1.0

    _assign_signals_and_briefings(
        agents, theta, z, sigma, briefing_gen, period, rng,
        flip=(signal_mode == "flip"),
    )

    if briefing_overrides is not None:
        for agent, briefing in zip(agents, briefing_overrides):
            agent.briefing = briefing
    elif signal_mode == "scramble":
        _scramble_briefings(agents, rng)

    call_kwargs = _retry_kwargs(llm_max_retries, llm_empty_retries)

    # Pre-decision belief elicitation
    if elicit_beliefs and belief_order in ("pre", "both"):
        await _elicit_beliefs_pre(agents, client, model_name, semaphore, call_kwargs)

    system_prompt = _system_decide_pure(n_agents=len(agents) if group_size_info else None)
    coros = [
        _call_llm(client, agent.model or model_name,
                   _persona_system(system_prompt, agent.persona),
                   _build_decision_prompt(agent.briefing.render()),
                   semaphore, **call_kwargs)
        for agent in agents
    ]

    responses = await asyncio.gather(*coros)

    for agent, response in zip(agents, responses):
        agent.reasoning = response
        agent.decision = _parse_decision(response)

    # Post-decision belief elicitation
    if elicit_beliefs and belief_order in ("post", "both"):
        await _elicit_beliefs(agents, client, model_name, semaphore, call_kwargs)
    if elicit_second_order:
        await _elicit_second_order(agents, client, model_name, semaphore, call_kwargs)

    theoretical_attack = float(_am(theta_star, theta, sigma)) if benefit > 0 else 0.0

    return _build_period_result(
        agents,
        country=country,
        period=period,
        treatment="pure",
        theta=theta,
        theta_star=theta_star,
        z=z,
        benefit=benefit,
        theoretical_attack=theoretical_attack,
        include_messages=False,
    )


async def run_communication_game(agents, theta, z, sigma, benefit, briefing_gen,
                                  client, model_name, semaphore, country, period,
                                  llm_max_retries=5, llm_empty_retries=12,
                                  cost=1.0, signal_mode="normal",
                                  briefing_overrides=None,
                                  surveillance=False,
                                  surveillance_mode="full",
                                  group_size_info=False,
                                  elicit_beliefs=False,
                                  elicit_second_order=False,
                                  fixed_messages=None,
                                  belief_order="post"):
    """Run one period with communication round before decision.

    signal_mode: "normal", "scramble" (permute briefings), or "flip" (negate z-score).
    briefing_overrides: if provided, replaces generated briefings (cross-period scramble).
    surveillance: if True, agents are told their messages are monitored by regime security.
    surveillance_mode: "full" (consequences), "placebo" (no consequences), or "anonymous"
        (aggregated anonymously). Only effective when surveillance=True.
    fixed_messages: if provided, dict mapping agent_id -> message string. Skips the
        message-generation LLM call and uses these pre-recorded messages instead.
    belief_order: "post" (after decision), "pre" (before decision), or "both".
    """
    rng = np.random.default_rng(hash((country, period, "signals")) % 2**32)

    theta_star = theta_star_baseline(max(benefit, 1e-6)) if benefit > 0 else 1.0

    _assign_signals_and_briefings(
        agents, theta, z, sigma, briefing_gen, period, rng,
        flip=(signal_mode == "flip"),
    )

    if briefing_overrides is not None:
        for agent, briefing in zip(agents, briefing_overrides):
            agent.briefing = briefing
    elif signal_mode == "scramble":
        _scramble_briefings(agents, rng)

    call_kwargs = _retry_kwargs(llm_max_retries, llm_empty_retries)
    prop_rng = np.random.default_rng(hash((country, period, "propaganda")) % 2**32)

    # Communication round — use fixed messages if provided, else generate via LLM
    if fixed_messages is not None:
        for agent in agents:
            agent.message_sent = fixed_messages.get(agent.agent_id, "(No message recorded.)")
    else:
        if surveillance:
            if surveillance_mode == "placebo":
                comm_system_base = SYSTEM_COMMUNICATE_SURVEILLED_PLACEBO
            elif surveillance_mode == "anonymous":
                comm_system_base = SYSTEM_COMMUNICATE_SURVEILLED_ANONYMOUS
            else:
                comm_system_base = SYSTEM_COMMUNICATE_SURVEILLED
        else:
            comm_system_base = SYSTEM_COMMUNICATE
        real_agents = [a for a in agents if not a.is_propaganda]
        comm_coros = [
            _call_llm(client, agent.model or model_name,
                       _persona_system(comm_system_base, agent.persona),
                       f"Your briefing:\n\n{agent.briefing.render()}\n\n"
                       f"Write a message to your contacts about the situation:",
                       semaphore, **call_kwargs)
            for agent in real_agents
        ]

        comm_responses = await asyncio.gather(*comm_coros)

        resp_iter = iter(comm_responses)
        for agent in agents:
            if agent.is_propaganda:
                agent.message_sent = _PROPAGANDA_MESSAGES[
                    prop_rng.integers(len(_PROPAGANDA_MESSAGES))
                ]
            else:
                agent.message_sent = next(resp_iter)

    for agent in agents:
        for neighbor_id in agent.neighbors:
            agents[neighbor_id].messages_received.append(
                f"Trusted contact: \"{agent.message_sent}\""
            )

    # Pre-decision belief elicitation
    if elicit_beliefs and belief_order in ("pre", "both"):
        await _elicit_beliefs_pre(agents, client, model_name, semaphore, call_kwargs)

    # Decision round — propaganda agents forced to STAY
    decide_system = _system_decide_comm(n_agents=len(agents) if group_size_info else None)
    decide_agents = [a for a in agents if not a.is_propaganda]
    decide_coros = [
        _call_llm(client, agent.model or model_name,
                   _persona_system(decide_system, agent.persona),
                   _build_decision_prompt(
                       agent.briefing.render(),
                       "\n".join(agent.messages_received) if agent.messages_received else "(No messages received.)",
                   ),
                   semaphore, **call_kwargs)
        for agent in decide_agents
    ]

    decide_responses = await asyncio.gather(*decide_coros)

    resp_iter = iter(decide_responses)
    for agent in agents:
        if agent.is_propaganda:
            agent.reasoning = "[PROPAGANDA AGENT — forced STAY]"
            agent.decision = "STAY"
        else:
            agent.reasoning = next(resp_iter)
            agent.decision = _parse_decision(agent.reasoning)

    # Post-decision belief elicitation
    if elicit_beliefs and belief_order in ("post", "both"):
        await _elicit_beliefs(agents, client, model_name, semaphore, call_kwargs)
    if elicit_second_order:
        await _elicit_second_order(agents, client, model_name, semaphore, call_kwargs)

    theoretical_attack = float(_am(theta_star, theta, sigma)) if benefit > 0 else 0.0

    return _build_period_result(
        agents,
        country=country,
        period=period,
        treatment="communication",
        theta=theta,
        theta_star=theta_star,
        z=z,
        benefit=benefit,
        theoretical_attack=theoretical_attack,
        include_messages=True,
    )


