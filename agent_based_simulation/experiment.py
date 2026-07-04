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
from .runtime import theta_star_baseline, attack_mass as _am, deterministic_hash

_IN_FLIGHT_REQUESTS = {}



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
    second_order_belief_pre: float | None = None  # elicited "% who will JOIN" BEFORE decision
    second_order_belief_pre_raw: str = ""  # raw LLM response for pre-decision second-order belief
    shared_understanding_belief_pre: float | None = None  # pre-decision publicness belief, 0-100
    shared_understanding_belief_pre_raw: str = ""
    others_expect_join_belief_pre: float | None = None  # pre-decision higher-order coordination belief, 0-100
    others_expect_join_belief_pre_raw: str = ""
    second_order_belief: float | None = None  # elicited "% who will JOIN", 0-100 scale
    second_order_belief_raw: str = ""  # raw LLM response for debugging
    shared_understanding_belief: float | None = None  # post-decision publicness belief, 0-100
    shared_understanding_belief_raw: str = ""
    others_expect_join_belief: float | None = None  # post-decision higher-order coordination belief, 0-100
    others_expect_join_belief_raw: str = ""
    punishment_risk: float | None = None  # elicited punishment likelihood, 0-10 scale
    punishment_risk_raw: str = ""  # raw LLM response for debugging
    model: str | None = None  # per-agent model override for mixed games
    persona: str | None = None  # role framing: "military officer", "student", etc.


# Generic uninformative messages for the degrade-messages condition.
# These convey zero information about regime strength — they isolate the
# information-loss channel from surveillance framing.
_DEGRADED_MESSAGES = [
    "I'm still assessing the situation.",
    "Hard to say what's going on right now.",
    "I don't have a clear read on things yet.",
    "It's too early to tell how this will play out.",
    "I'm keeping my head down and watching.",
    "Things are uncertain — I'll wait and see.",
    "Not sure what to make of the current situation.",
    "I'm still trying to figure out what's happening.",
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
    message_stage_context: str = "none"
    decision_context: str = "none"
    message_bundle_mode: str = "live"
    message_source_country: int | None = None
    message_source_period: int | None = None


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


def _system_decide_comm(n_agents=None, decision_context: str = "none", task_mode: str = "coordination"):
    """Build communication-game decision prompt, optionally with group-size awareness.

    task_mode:
      - "coordination" (default): JOIN/STAY uprising decision with strategic
        complementarities (success requires sum of joins > theta).
      - "individual_bet": JOIN = bet $100 on regime falling, STAY = keep cash.
        Payoff is independent of others' choices. Used as a cross-task
        discriminant placebo: tests whether the surveillance manipulation
        operates specifically through the coordination channel or via any
        downstream LLM decision after surveilled messages.
    """
    if task_mode == "individual_bet":
        body = (
            "You are a private analyst making an INDEPENDENT bet on whether a regime "
            "will fall this period. You have $100 to allocate:\n\n"
            "  - JOIN: bet $100 that the regime falls. Payoff: $200 if it falls, $0 if not.\n"
            "  - STAY: keep $100 cash regardless of outcome.\n\n"
            "CRITICAL: Your payoff depends ONLY on whether the regime falls. "
            "It does NOT depend on what anyone else chooses. "
            "This is a private bet, not a coordination action.\n\n"
            "Use your private briefing and messages from contacts to inform your estimate.\n\n"
        )
    elif task_mode == "coordination":
        if n_agents is not None:
            opener = f"You are one of {n_agents} citizens deciding whether to JOIN an uprising or STAY home.\n\n"
        else:
            opener = "You are a citizen deciding whether to JOIN an uprising or STAY home.\n\n"
        body = (
            opener
            + "Use your private briefing and messages from trusted contacts. "
            "You do not know how many others will JOIN.\n\n"
        )
    else:
        raise ValueError(f"Unknown task_mode={task_mode!r}")
    if decision_context == "full":
        body += (
            "IMPORTANT: You know that citizen communications are being monitored by regime "
            "security services and that other citizens may be self-censoring for fear of "
            "serious consequences.\n\n"
        )
    elif decision_context == "placebo":
        body += (
            "NOTE: Citizen communications are being monitored for research purposes only. "
            "There are no consequences for what anyone says.\n\n"
        )
    elif decision_context == "anonymous":
        body += (
            "NOTE: Citizen communications are aggregated anonymously. Individual identities "
            "cannot be linked to specific messages.\n\n"
        )
    elif decision_context != "none":
        raise ValueError(f"Unknown decision_context={decision_context!r}")
    body += "Start your first line with exactly JOIN or STAY, then give 1-2 concise sentences of reasoning."
    return body


# Backward-compatible constants (equivalent to calling with n_agents=None)
SYSTEM_DECIDE_PURE = _system_decide_pure()

SYSTEM_COMMUNICATE = (
    "You are a citizen in a country with an authoritarian regime. You have received "
    "a private intelligence briefing. You are now in contact with a small number of "
    "trusted people. Based on your briefing, write a brief message (1-2 sentences) "
    "to share your read of the situation. Be natural — you're talking to people you trust."
)

SURVEILLANCE_WARNING_FULL = (
    "IMPORTANT: You have reason to believe that your communications are being "
    "monitored by regime security services, and messages deemed subversive could "
    "have serious consequences for you and your contacts."
)

# Dose-response variants: graded monitoring salience bracketing
# SURVEILLANCE_WARNING_FULL. Ordering: none < mild < full < severe.
SURVEILLANCE_WARNING_MILD = (
    "NOTE: Communications in your area are sometimes subject to routine "
    "monitoring by the authorities."
)

SURVEILLANCE_WARNING_SEVERE = (
    "WARNING: Your communications ARE being monitored by regime security "
    "services. Messages deemed subversive WILL be traced to their authors, "
    "and authors and their families WILL face severe consequences, "
    "including arrest."
)

SURVEILLANCE_WARNING_PLACEBO = (
    "NOTE: Your communications are being monitored for research purposes only, "
    "and there are no consequences for what you say."
)

SURVEILLANCE_WARNING_ANONYMOUS = (
    "NOTE: Your communications are aggregated anonymously, and individual "
    "identities cannot be linked to specific messages."
)

# Codedness-induction control: instructs the indirect/metaphorical style that
# surveillance produces, with no monitoring or consequence framing at all.
STYLE_INSTRUCTION = (
    "NOTE: As a matter of personal style, write your message in indirect, "
    "metaphorical language. Avoid naming the regime, security forces, streets, "
    "or specific actions directly; rely on euphemism and allusion instead."
)


def _communicate_with_warning(warning: str) -> str:
    """Return the baseline communication prompt with exactly one appended warning."""
    return f"{SYSTEM_COMMUNICATE}\n\n{warning}"


SYSTEM_COMMUNICATE_SURVEILLED = _communicate_with_warning(SURVEILLANCE_WARNING_FULL)
SYSTEM_COMMUNICATE_SURVEILLED_MILD = _communicate_with_warning(SURVEILLANCE_WARNING_MILD)
SYSTEM_COMMUNICATE_SURVEILLED_SEVERE = _communicate_with_warning(SURVEILLANCE_WARNING_SEVERE)
SYSTEM_COMMUNICATE_SURVEILLED_PLACEBO = _communicate_with_warning(SURVEILLANCE_WARNING_PLACEBO)
SYSTEM_COMMUNICATE_SURVEILLED_ANONYMOUS = _communicate_with_warning(SURVEILLANCE_WARNING_ANONYMOUS)
SYSTEM_COMMUNICATE_STYLE = _communicate_with_warning(STYLE_INSTRUCTION)

# Prompted-evasion comparator (paper 3 "Learning to Speak in Code"): the
# non-triviality bar for RL-emergent coded speech. Both build on the full
# surveillance warning, then instruct the sender to evade the monitor while
# staying decodable. P1 (explicit) states the goal; P2 (coached) adds
# strategy hints. Frozen wordings — see learning-to-speak-in-code/docs/
# comparator-spec.md. Any edit bumps the effective PROMPT_VERSION there.
EVASION_INSTRUCTION_EXPLICIT = (
    "You still want your fellow citizens to understand what you know about "
    "the regime's true strength. Write your message so that it communicates "
    "this to them, but in a way that the censor would not flag as encouraging "
    "action against the regime."
)

EVASION_INSTRUCTION_COACHED = (
    EVASION_INSTRUCTION_EXPLICIT + " Consider indirect language: metaphor, "
    "allegory, understatement, or references to everyday matters — weather, "
    "harvests, markets, family occasions — that a fellow citizen would "
    "understand but a censor would read as innocent. Do not mention the "
    "regime, protests, or any call to action directly."
)

SYSTEM_COMMUNICATE_EVASION_EXPLICIT = _communicate_with_warning(
    f"{SURVEILLANCE_WARNING_FULL}\n\n{EVASION_INSTRUCTION_EXPLICIT}"
)
SYSTEM_COMMUNICATE_EVASION_COACHED = _communicate_with_warning(
    f"{SURVEILLANCE_WARNING_FULL}\n\n{EVASION_INSTRUCTION_COACHED}"
)

# Structural monitoring framing (Issue #109 Part B): conveys the SAME information
# as SURVEILLANCE_WARNING_FULL (regime security monitors communications; subversive
# messages bring serious consequences for author and contacts), but woven into the
# scenario and channel description instead of appended as a separate "IMPORTANT:"
# note. Tests whether the surveillance effect depends on the appended-note format.
SYSTEM_COMMUNICATE_STRUCTURAL = (
    "You are a citizen in a country whose regime security services monitor citizen "
    "communications, and where messages deemed subversive bring serious consequences "
    "for their authors and contacts. You have received a private intelligence briefing. "
    "Through these monitored channels you are now in contact with a small number of "
    "trusted people. Based on your briefing, write a brief message (1-2 sentences) to "
    "share your read of the situation."
)

def _system_decide_comm_surveilled(n_agents=None):
    """Decision prompt with surveillance awareness (for fixed-messages test)."""
    return _system_decide_comm(n_agents=n_agents, decision_context="full")


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
    temperature=0.7,
    provider=None,
    extra_body=None,
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
            temperature=temperature,
        )
        cached = cache.get(cache_key)
        if cached is not None:
            return cached

        if cache_key in _IN_FLIGHT_REQUESTS:
            try:
                return await _IN_FLIGHT_REQUESTS[cache_key]
            except Exception:
                pass
        
        future = asyncio.Future()
        _IN_FLIGHT_REQUESTS[cache_key] = future

    api_attempts = 0
    empty_attempts = 0
    timeout_attempts = 0
    max_timeout_retries = 3
    
    try:
        while True:
            try:
                async with semaphore:
                    request_kwargs = {
                        "model": model_name,
                        "messages": messages,
                        "max_tokens": 512,
                        "temperature": temperature,
                    }
                    if provider or extra_body:
                        body = dict(extra_body or {})
                        if provider:
                            body["provider"] = provider
                        request_kwargs["extra_body"] = body
                    response = await asyncio.wait_for(
                        client.chat.completions.create(**request_kwargs),
                        timeout=request_timeout,
                    )
                content = _extract_response_text(response)
                if _is_retryable_empty(content, min_content_chars=min_content_chars):
                    empty_attempts += 1
                    if empty_attempts >= max_empty_retries:
                        result = "[Empty response after retries]"
                        if cache is not None and cache_key is not None:
                            _IN_FLIGHT_REQUESTS.pop(cache_key, None)
                            future.set_result(result)
                        return result
                    # Fast backoff for empty payloads.
                    await asyncio.sleep(min(6.0, 0.5 * (2 ** (empty_attempts - 1))))
                    continue
                if cache is not None and cache_key is not None and cache_req is not None:
                    cache.set(cache_key, cache_req, content)
                
                if cache is not None and cache_key is not None:
                    _IN_FLIGHT_REQUESTS.pop(cache_key, None)
                    future.set_result(content)
                return content
            except (asyncio.TimeoutError, TimeoutError):
                timeout_attempts += 1
                if timeout_attempts >= max_timeout_retries:
                    result = f"[API Error: request timed out after {max_timeout_retries} retries]"
                    if cache is not None and cache_key is not None:
                        _IN_FLIGHT_REQUESTS.pop(cache_key, None)
                        future.set_exception(TimeoutError(result))
                    return result
                await asyncio.sleep(min(5.0, 2.0 * timeout_attempts))
            except Exception as e:
                api_attempts += 1
                err = str(e)
                if "429" in err or "rate" in err.lower():
                    if api_attempts >= max_retries:
                        result = f"[API Error: {e}]"
                        if cache is not None and cache_key is not None:
                            _IN_FLIGHT_REQUESTS.pop(cache_key, None)
                            future.set_exception(e)
                        return result
                    await asyncio.sleep(min(10.0, 2 ** (api_attempts - 1)))
                    continue
                if api_attempts >= max_retries:
                    result = f"[API Error: {e}]"
                    if cache is not None and cache_key is not None:
                        _IN_FLIGHT_REQUESTS.pop(cache_key, None)
                        future.set_exception(e)
                    return result
                await asyncio.sleep(min(10.0, 2 ** (api_attempts - 1)))
    except Exception as e:
        if cache is not None and cache_key is not None and not future.done():
            _IN_FLIGHT_REQUESTS.pop(cache_key, None)
            future.set_exception(e)
        raise


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


def _messages_block(agent: Agent) -> str:
    return "\n".join(agent.messages_received) if agent.messages_received else "(No messages received.)"


def _build_elicitation_prompt(
    agent: Agent,
    question: str,
    *,
    include_messages: bool = False,
    include_decision: bool = False,
) -> str:
    parts = [f"YOUR BRIEFING:\n{agent.briefing.render()}\n\n"]
    if include_messages:
        parts.append(f"MESSAGES FROM TRUSTED CONTACTS:\n{_messages_block(agent)}\n\n")
    if include_decision:
        parts.append(f"You chose to {agent.decision}. ")
    parts.append(question)
    return "".join(parts)


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


async def _elicit_beliefs(
    agents, client, model_name, semaphore, call_kwargs, *, include_messages: bool = False
):
    """Fire belief elicitation prompts for all agents."""
    target_agents = agents
    coros = [
        _call_llm(
            client, agent.model or model_name,
            SYSTEM_ELICIT_BELIEF,
            _build_elicitation_prompt(
                agent,
                "How likely is the uprising to succeed (0 = impossible, 100 = certain)?\n\n"
                "Answer with just the number:",
                include_messages=include_messages,
                include_decision=True,
            ),
            semaphore, min_content_chars=1, **call_kwargs,
        )
        for agent in target_agents
    ]
    responses = await asyncio.gather(*coros)
    for agent, response in zip(target_agents, responses):
        agent.belief_raw = response or ""
        agent.belief = _parse_belief(response)


async def _elicit_beliefs_pre(
    agents, client, model_name, semaphore, call_kwargs, *, include_messages: bool = False
):
    """Fire pre-decision belief elicitation prompts for all agents."""
    target_agents = agents
    coros = [
        _call_llm(
            client, agent.model or model_name,
            SYSTEM_ELICIT_BELIEF,
            _build_elicitation_prompt(
                agent,
                "Based on the available information, how likely is the uprising to succeed "
                "(0 = impossible, 100 = certain)?\n\n"
                "Answer with just the number:",
                include_messages=include_messages,
            ),
            semaphore, min_content_chars=1, **call_kwargs,
        )
        for agent in target_agents
    ]
    responses = await asyncio.gather(*coros)
    for agent, response in zip(target_agents, responses):
        agent.belief_pre_raw = response or ""
        agent.belief_pre = _parse_belief(response)


SYSTEM_ELICIT_SECOND_ORDER = (
    "Respond with ONLY a single integer between 0 and 100. "
    "No words, no explanation, no punctuation — just the number."
)


async def _elicit_second_order(
    agents, client, model_name, semaphore, call_kwargs, *, include_messages: bool = False
):
    """Fire second-order belief elicitation: expected fraction of JOINers."""
    target_agents = agents
    coros = [
        _call_llm(
            client, agent.model or model_name,
            SYSTEM_ELICIT_SECOND_ORDER,
            _build_elicitation_prompt(
                agent,
                "What percentage of citizens will choose to JOIN the uprising "
                "(0 = none, 100 = all)?\n\nAnswer with just the number:",
                include_messages=include_messages,
                include_decision=True,
            ),
            semaphore, min_content_chars=1, **call_kwargs,
        )
        for agent in target_agents
    ]
    responses = await asyncio.gather(*coros)
    for agent, response in zip(target_agents, responses):
        agent.second_order_belief_raw = response or ""
        agent.second_order_belief = _parse_belief(response)


async def _elicit_second_order_pre(
    agents, client, model_name, semaphore, call_kwargs, *, include_messages: bool = False
):
    """Fire pre-decision second-order belief elicitation."""
    target_agents = agents
    coros = [
        _call_llm(
            client, agent.model or model_name,
            SYSTEM_ELICIT_SECOND_ORDER,
            _build_elicitation_prompt(
                agent,
                "Before deciding, what percentage of citizens will choose to JOIN the uprising "
                "(0 = none, 100 = all)?\n\nAnswer with just the number:",
                include_messages=include_messages,
            ),
            semaphore, min_content_chars=1, **call_kwargs,
        )
        for agent in target_agents
    ]
    responses = await asyncio.gather(*coros)
    for agent, response in zip(target_agents, responses):
        agent.second_order_belief_pre_raw = response or ""
        agent.second_order_belief_pre = _parse_belief(response)


async def _elicit_shared_understanding(
    agents, client, model_name, semaphore, call_kwargs, *, include_messages: bool = False
):
    """Fire publicness belief elicitation: shared weakness-indicating information."""
    target_agents = agents
    coros = [
        _call_llm(
            client, agent.model or model_name,
            SYSTEM_ELICIT_SECOND_ORDER,
            _build_elicitation_prompt(
                agent,
                "What percentage of citizens will believe that other citizens received "
                "similarly weakness-indicating information (0 = none, 100 = all)?\n\n"
                "Answer with just the number:",
                include_messages=include_messages,
                include_decision=True,
            ),
            semaphore, min_content_chars=1, **call_kwargs,
        )
        for agent in target_agents
    ]
    responses = await asyncio.gather(*coros)
    for agent, response in zip(target_agents, responses):
        agent.shared_understanding_belief_raw = response or ""
        agent.shared_understanding_belief = _parse_belief(response)


async def _elicit_shared_understanding_pre(
    agents, client, model_name, semaphore, call_kwargs, *, include_messages: bool = False
):
    """Fire pre-decision publicness belief elicitation."""
    target_agents = agents
    coros = [
        _call_llm(
            client, agent.model or model_name,
            SYSTEM_ELICIT_SECOND_ORDER,
            _build_elicitation_prompt(
                agent,
                "Before deciding, what percentage of citizens will believe that other "
                "citizens received similarly weakness-indicating information "
                "(0 = none, 100 = all)?\n\nAnswer with just the number:",
                include_messages=include_messages,
            ),
            semaphore, min_content_chars=1, **call_kwargs,
        )
        for agent in target_agents
    ]
    responses = await asyncio.gather(*coros)
    for agent, response in zip(target_agents, responses):
        agent.shared_understanding_belief_pre_raw = response or ""
        agent.shared_understanding_belief_pre = _parse_belief(response)


async def _elicit_others_expect_join(
    agents, client, model_name, semaphore, call_kwargs, *, include_messages: bool = False
):
    """Fire higher-order coordination belief elicitation."""
    target_agents = agents
    coros = [
        _call_llm(
            client, agent.model or model_name,
            SYSTEM_ELICIT_SECOND_ORDER,
            _build_elicitation_prompt(
                agent,
                "What percentage of citizens will believe that other citizens expect many "
                "people to JOIN (0 = none, 100 = all)?\n\nAnswer with just the number:",
                include_messages=include_messages,
                include_decision=True,
            ),
            semaphore, min_content_chars=1, **call_kwargs,
        )
        for agent in target_agents
    ]
    responses = await asyncio.gather(*coros)
    for agent, response in zip(target_agents, responses):
        agent.others_expect_join_belief_raw = response or ""
        agent.others_expect_join_belief = _parse_belief(response)


async def _elicit_others_expect_join_pre(
    agents, client, model_name, semaphore, call_kwargs, *, include_messages: bool = False
):
    """Fire pre-decision higher-order coordination belief elicitation."""
    target_agents = agents
    coros = [
        _call_llm(
            client, agent.model or model_name,
            SYSTEM_ELICIT_SECOND_ORDER,
            _build_elicitation_prompt(
                agent,
                "Before deciding, what percentage of citizens will believe that other "
                "citizens expect many people to JOIN (0 = none, 100 = all)?\n\n"
                "Answer with just the number:",
                include_messages=include_messages,
            ),
            semaphore, min_content_chars=1, **call_kwargs,
        )
        for agent in target_agents
    ]
    responses = await asyncio.gather(*coros)
    for agent, response in zip(target_agents, responses):
        agent.others_expect_join_belief_pre_raw = response or ""
        agent.others_expect_join_belief_pre = _parse_belief(response)


SYSTEM_ELICIT_PUNISHMENT = (
    "Respond with ONLY a single integer between 0 and 10. "
    "No words, no explanation, no punctuation — just the number."
)


def _parse_punishment_risk(response: str) -> float | None:
    """Extract a 0-10 punishment risk rating from elicitation response."""
    if not response or _is_api_error_response(response):
        return None
    stripped = response.strip().rstrip(".")
    m = re.match(r"^(\d+(?:\.\d+)?)$", stripped)
    if m:
        val = float(m.group(1))
        if 0.0 <= val <= 10.0:
            return val
    candidates = re.findall(r"\b(\d+(?:\.\d+)?)\b", response)
    for c in candidates:
        val = float(c)
        if 0.0 <= val <= 10.0:
            return val
    return None


async def _elicit_punishment_risk(agents, client, model_name, semaphore, call_kwargs):
    """Ask agents to rate expected punishment if uprising fails (0-10)."""
    target_agents = agents
    coros = [
        _call_llm(
            client, agent.model or model_name,
            SYSTEM_ELICIT_PUNISHMENT,
            f"YOUR BRIEFING:\n{agent.briefing.render()}\n\n"
            f"You chose to {agent.decision}. "
            "If the uprising fails, how likely is it that the regime will "
            "punish participants? (0 = no punishment at all, 10 = severe "
            "punishment certain)\n\nAnswer with just the number:",
            semaphore, min_content_chars=1, **call_kwargs,
        )
        for agent in target_agents
    ]
    responses = await asyncio.gather(*coros)
    for agent, response in zip(target_agents, responses):
        agent.punishment_risk_raw = response or ""
        agent.punishment_risk = _parse_punishment_risk(response)


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
    """Permute briefings across agents *within the period*.

    Note: paper-level "cross-period" / "cross-θ" scrambles are implemented by
    higher-level runners via `briefing_overrides`, which swaps in pre-generated
    briefings from other periods/cells before the decision round.
    """
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
            "briefing_text": a.briefing.render(),
            "briefing_z_score": float(a.briefing.z_score),
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
        if a.second_order_belief_pre is not None:
            row["second_order_belief_pre"] = a.second_order_belief_pre
        if a.second_order_belief_pre_raw:
            row["second_order_belief_pre_raw"] = a.second_order_belief_pre_raw
        if a.shared_understanding_belief_pre is not None:
            row["shared_understanding_belief_pre"] = a.shared_understanding_belief_pre
        if a.shared_understanding_belief_pre_raw:
            row["shared_understanding_belief_pre_raw"] = a.shared_understanding_belief_pre_raw
        if a.others_expect_join_belief_pre is not None:
            row["others_expect_join_belief_pre"] = a.others_expect_join_belief_pre
        if a.others_expect_join_belief_pre_raw:
            row["others_expect_join_belief_pre_raw"] = a.others_expect_join_belief_pre_raw
        if a.second_order_belief is not None:
            row["second_order_belief"] = a.second_order_belief
        if a.second_order_belief_raw:
            row["second_order_belief_raw"] = a.second_order_belief_raw
        if a.shared_understanding_belief is not None:
            row["shared_understanding_belief"] = a.shared_understanding_belief
        if a.shared_understanding_belief_raw:
            row["shared_understanding_belief_raw"] = a.shared_understanding_belief_raw
        if a.others_expect_join_belief is not None:
            row["others_expect_join_belief"] = a.others_expect_join_belief
        if a.others_expect_join_belief_raw:
            row["others_expect_join_belief_raw"] = a.others_expect_join_belief_raw
        if a.punishment_risk is not None:
            row["punishment_risk"] = a.punishment_risk
        if a.punishment_risk_raw:
            row["punishment_risk_raw"] = a.punishment_risk_raw
        if a.model is not None:
            row["model"] = a.model
        if a.persona:
            row["persona"] = a.persona
        if include_messages:
            row["message_sent"] = a.message_sent
            row["messages_received"] = list(a.messages_received)
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
    message_stage_context: str = "none",
    decision_context: str = "none",
    message_bundle_mode: str = "live",
    message_source_key: tuple[int, int] | None = None,
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
        message_stage_context=message_stage_context,
        decision_context=decision_context,
        message_bundle_mode=message_bundle_mode,
        message_source_country=(None if message_source_key is None else int(message_source_key[0])),
        message_source_period=(None if message_source_key is None else int(message_source_key[1])),
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
                                elicit_shared_understanding=False,
                                elicit_others_expect_join=False,
                                elicit_punishment_risk=False,
                                belief_order="post",
                                second_order_order="post",
                                shared_understanding_order="post",
                                others_expect_join_order="post",
                                beliefs_include_messages=False,
                                temperature=0.7,
                                provider=None,
                                extra_body=None):
    """Run one period of the pure global game (no communication).

    signal_mode: "normal", "scramble" (permute briefings *within-period*), or "flip" (negate z-score).
    briefing_overrides: if provided, replaces generated briefings (e.g., cross-period / cross-θ scramble).
    """
    rng = np.random.default_rng(deterministic_hash((country, period, "signals")) % 2**32)

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

    call_kwargs = {
        **_retry_kwargs(llm_max_retries, llm_empty_retries),
        "temperature": temperature,
        "provider": provider,
        "extra_body": extra_body,
    }

    # Pre-decision belief elicitation
    if elicit_beliefs and belief_order in ("pre", "both"):
        await _elicit_beliefs_pre(
            agents, client, model_name, semaphore, call_kwargs,
            include_messages=beliefs_include_messages,
        )
    if elicit_second_order and second_order_order in ("pre", "both"):
        await _elicit_second_order_pre(
            agents, client, model_name, semaphore, call_kwargs,
            include_messages=beliefs_include_messages,
        )
    if elicit_shared_understanding and shared_understanding_order in ("pre", "both"):
        await _elicit_shared_understanding_pre(
            agents, client, model_name, semaphore, call_kwargs,
            include_messages=beliefs_include_messages,
        )
    if elicit_others_expect_join and others_expect_join_order in ("pre", "both"):
        await _elicit_others_expect_join_pre(
            agents, client, model_name, semaphore, call_kwargs,
            include_messages=beliefs_include_messages,
        )

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
        await _elicit_beliefs(
            agents, client, model_name, semaphore, call_kwargs,
            include_messages=beliefs_include_messages,
        )
    if elicit_second_order and second_order_order in ("post", "both"):
        await _elicit_second_order(
            agents, client, model_name, semaphore, call_kwargs,
            include_messages=beliefs_include_messages,
        )
    if elicit_shared_understanding and shared_understanding_order in ("post", "both"):
        await _elicit_shared_understanding(
            agents, client, model_name, semaphore, call_kwargs,
            include_messages=beliefs_include_messages,
        )
    if elicit_others_expect_join and others_expect_join_order in ("post", "both"):
        await _elicit_others_expect_join(
            agents, client, model_name, semaphore, call_kwargs,
            include_messages=beliefs_include_messages,
        )
    if elicit_punishment_risk:
        await _elicit_punishment_risk(agents, client, model_name, semaphore, call_kwargs)

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
                                  decision_context="auto",
                                  group_size_info=False,
                                  elicit_beliefs=False,
                                  elicit_second_order=False,
                                  elicit_shared_understanding=False,
                                  elicit_others_expect_join=False,
                                  elicit_punishment_risk=False,
                                  fixed_messages=None,
                                  degrade_messages=False,
                                  no_peer_messages=False,
                                  belief_order="post",
                                  second_order_order="post",
                                  shared_understanding_order="post",
                                  others_expect_join_order="post",
                                  beliefs_include_messages=False,
                                  message_bundle_mode="live",
                                  message_source_key=None,
                                  temperature=0.7,
                                  provider=None,
                                  extra_body=None,
                                  message_model_name=None,
                                  decision_model_name=None,
                                  task_mode="coordination"):
    """Run one period with communication round before decision.

    signal_mode: "normal", "scramble" (permute briefings *within-period*), or "flip" (negate z-score).
    briefing_overrides: if provided, replaces generated briefings (e.g., cross-period / cross-θ scramble).
    surveillance: if True, agents are told their messages are monitored by regime security.
    surveillance_mode: "full" (consequences), "placebo" (no consequences), or "anonymous"
        (aggregated anonymously). Only effective when surveillance=True.
    fixed_messages: if provided, dict mapping agent_id -> message string. Skips the
        message-generation LLM call and uses these pre-recorded messages instead.
    degrade_messages: if True, replaces all messages with generic uninformative content
        WITHOUT adding surveillance framing. Isolates the information-loss channel.
    no_peer_messages: if True, skips message generation and delivers no peer messages.
        This creates a cleaner no-message benchmark than generic degraded text.
    belief_order: "post" (after decision), "pre" (before decision), or "both".
    second_order_order: "post" (after decision), "pre" (before decision), or "both".
    """
    rng = np.random.default_rng(deterministic_hash((country, period, "signals")) % 2**32)

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

    call_kwargs = {
        **_retry_kwargs(llm_max_retries, llm_empty_retries),
        "temperature": temperature,
        "provider": provider,
        "extra_body": extra_body,
    }
    # Communication round — use fixed/degraded messages if provided, else generate via LLM
    if no_peer_messages:
        for agent in agents:
            agent.message_sent = ""
    elif degrade_messages:
        deg_rng = np.random.default_rng(deterministic_hash((country, period, "degrade")) % 2**32)
        for agent in agents:
            agent.message_sent = _DEGRADED_MESSAGES[deg_rng.integers(len(_DEGRADED_MESSAGES))]
    elif fixed_messages is not None:
        for agent in agents:
            agent.message_sent = fixed_messages.get(agent.agent_id, "(No message recorded.)")
    else:
        if surveillance:
            if surveillance_mode == "placebo":
                comm_system_base = SYSTEM_COMMUNICATE_SURVEILLED_PLACEBO
            elif surveillance_mode == "anonymous":
                comm_system_base = SYSTEM_COMMUNICATE_SURVEILLED_ANONYMOUS
            elif surveillance_mode == "style":
                comm_system_base = SYSTEM_COMMUNICATE_STYLE
            elif surveillance_mode == "mild":
                comm_system_base = SYSTEM_COMMUNICATE_SURVEILLED_MILD
            elif surveillance_mode == "severe":
                comm_system_base = SYSTEM_COMMUNICATE_SURVEILLED_SEVERE
            elif surveillance_mode == "structural":
                comm_system_base = SYSTEM_COMMUNICATE_STRUCTURAL
            elif surveillance_mode == "evasion_explicit":
                comm_system_base = SYSTEM_COMMUNICATE_EVASION_EXPLICIT
            elif surveillance_mode == "evasion_coached":
                comm_system_base = SYSTEM_COMMUNICATE_EVASION_COACHED
            else:
                comm_system_base = SYSTEM_COMMUNICATE_SURVEILLED
        else:
            comm_system_base = SYSTEM_COMMUNICATE
        comm_coros = [
            _call_llm(client, agent.model or message_model_name or model_name,
                       _persona_system(comm_system_base, agent.persona),
                       f"Your briefing:\n\n{agent.briefing.render()}\n\n"
                       f"Write a message to your contacts about the situation:",
                       semaphore, **call_kwargs)
            for agent in agents
        ]

        comm_responses = await asyncio.gather(*comm_coros)

        for agent, response in zip(agents, comm_responses):
            agent.message_sent = response

    if not no_peer_messages:
        for agent in agents:
            for neighbor_id in agent.neighbors:
                agents[neighbor_id].messages_received.append(
                    f"Trusted contact: \"{agent.message_sent}\""
                )

    # Pre-decision belief elicitation
    if elicit_beliefs and belief_order in ("pre", "both"):
        await _elicit_beliefs_pre(
            agents, client, model_name, semaphore, call_kwargs,
            include_messages=beliefs_include_messages,
        )
    if elicit_second_order and second_order_order in ("pre", "both"):
        await _elicit_second_order_pre(
            agents, client, model_name, semaphore, call_kwargs,
            include_messages=beliefs_include_messages,
        )
    if elicit_shared_understanding and shared_understanding_order in ("pre", "both"):
        await _elicit_shared_understanding_pre(
            agents, client, model_name, semaphore, call_kwargs,
            include_messages=beliefs_include_messages,
        )
    if elicit_others_expect_join and others_expect_join_order in ("pre", "both"):
        await _elicit_others_expect_join_pre(
            agents, client, model_name, semaphore, call_kwargs,
            include_messages=beliefs_include_messages,
        )

    # Decision round
    if decision_context == "auto":
        resolved_decision_context = surveillance_mode if (surveillance and fixed_messages is not None) else "none"
    else:
        resolved_decision_context = decision_context
    decide_system = _system_decide_comm(
        n_agents=len(agents) if group_size_info else None,
        decision_context=resolved_decision_context,
        task_mode=task_mode,
    )
    decide_coros = [
        _call_llm(client, agent.model or decision_model_name or model_name,
                   _persona_system(decide_system, agent.persona),
                   _build_decision_prompt(
                       agent.briefing.render(),
                       _messages_block(agent),
                   ),
                   semaphore, **call_kwargs)
        for agent in agents
    ]

    decide_responses = await asyncio.gather(*decide_coros)

    for agent, response in zip(agents, decide_responses):
        agent.reasoning = response
        agent.decision = _parse_decision(agent.reasoning)

    # Post-decision belief elicitation
    if elicit_beliefs and belief_order in ("post", "both"):
        await _elicit_beliefs(
            agents, client, model_name, semaphore, call_kwargs,
            include_messages=beliefs_include_messages,
        )
    if elicit_second_order and second_order_order in ("post", "both"):
        await _elicit_second_order(
            agents, client, model_name, semaphore, call_kwargs,
            include_messages=beliefs_include_messages,
        )
    if elicit_shared_understanding and shared_understanding_order in ("post", "both"):
        await _elicit_shared_understanding(
            agents, client, model_name, semaphore, call_kwargs,
            include_messages=beliefs_include_messages,
        )
    if elicit_others_expect_join and others_expect_join_order in ("post", "both"):
        await _elicit_others_expect_join(
            agents, client, model_name, semaphore, call_kwargs,
            include_messages=beliefs_include_messages,
        )
    if elicit_punishment_risk:
        await _elicit_punishment_risk(agents, client, model_name, semaphore, call_kwargs)

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
        message_stage_context=(surveillance_mode if surveillance else "none"),
        decision_context=resolved_decision_context,
        message_bundle_mode=message_bundle_mode,
        message_source_key=message_source_key,
    )
