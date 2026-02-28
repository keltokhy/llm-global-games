"""
Evidence mosaic briefing generator.

Converts a continuous private signal x_i into a structured intelligence
briefing with multiple qualitative cues. The composition of many small
graded choices recovers effective continuity — the "dithering" principle.

Three latent sliders derived from x_i:
  A. Direction  (weak ↔ strong) — monotone in x_i
  B. Clarity    (ambiguous ↔ clear) — U-shaped, lowest near cutoff
  C. Coordination climate (quiet ↔ open) — monotone in x_i

Each briefing has a fixed schema:
  - Bottom line (one sentence)
  - Observations (8 bullets, each with domain/valence/intensity/corroboration)
  - What's unclear (2 bullets)
  - Atmosphere (one sentence)
"""

import numpy as np
from dataclasses import dataclass
from .runtime import deterministic_hash

DEFAULT_BOTTOMLINE_CUTS = (0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85)
DEFAULT_UNCLEAR_CUTS = (0.18, 0.33, 0.48, 0.62, 0.77)
DEFAULT_COORDINATION_CUTS = (0.12, 0.25, 0.42, 0.58, 0.75)
DEFAULT_COORDINATION_BLEND_PROB = 0.60
DEFAULT_LANGUAGE_VARIANT = "baseline"
SUPPORTED_LANGUAGE_VARIANTS = (
    "legacy",
    "baseline_min",
    "baseline",
    "baseline_assess",
    "baseline_full",
    "cable",
    "journalistic",
)

# ---------------------------------------------------------------------------
# Domain indices: the 8 observation domains in generation order
# ---------------------------------------------------------------------------
DOMAIN_NAMES = [
    "elite_cohesion",           # 0
    "security_forces",          # 1
    "money_and_logistics",      # 2
    "street_mood",              # 3
    "information_control",      # 4
    "personal_observations",    # 5
    "diplomatic_signals",       # 6
    "institutional_functioning",# 7
]
COORDINATION_DOMAINS = {3, 5}       # street_mood, personal_observations
STATE_CAPACITY_DOMAINS = {0, 1, 4, 7}  # elite, security, info_control, institutional

# ---------------------------------------------------------------------------
# Phrase ladders: ordered sequences from weak → strong expression
# ---------------------------------------------------------------------------

CORROBORATION_LADDER = [
    "a single thread, hard to verify",
    "one source, unconfirmed and possibly unreliable",
    "an isolated mention from one source",
    "two sources, but they may have heard the same thing",
    "a couple of contacts mentioning it independently",
    "a few people saying similar things, though none with direct knowledge",
    "recurring mention from separate contacts",
    "the same story surfacing through unrelated networks",
    "several sources saying similar things without coordination",
    "a pattern emerging across different circles",
    "independent echoes across channels",
    "consistent reporting from sources who don't know each other",
    "multiple channels lining up consistently",
    "broad agreement across reliable and less reliable sources alike",
    "near-unanimous confirmation from every reliable source",
    "overwhelming convergence — every channel telling the same story independently",
]

INTENSITY_LADDER = [
    "barely perceptible",
    "hairline",
    "faint but real",
    "slight but persistent",
    "noticeable",
    "growing harder to dismiss",
    "unmistakable",
    "deepening",
    "pronounced",
    "open",
    "acute",
    "intensifying rapidly",
    "severe",
    "approaching a breaking point",
    "irreversible",
    "past the point of concealment",
]

MODALITY_LADDER = [
    "might, under certain conditions,",
    "might",
    "could conceivably",
    "could",
    "seems to",
    "appears to",
    "tends to indicate",
    "suggests",
    "points to",
    "consistently points to",
    "strongly suggests",
    "makes it difficult to avoid concluding that",
    "leaves little doubt that",
    "makes it nearly certain that",
    "is hard to escape the conclusion that",
    "removes any serious doubt that",
]

MOMENTUM_LADDER = [
    "settling back toward normal",
    "decelerating and losing whatever energy it had",
    "settling and stabilizing",
    "cooling off, though not yet cold",
    "holding roughly steady",
    "flat, but with a sense of waiting",
    "neither advancing nor retreating — suspended",
    "drifting without clear direction",
    "beginning to tilt, though the direction could still reverse",
    "tilting in a way that could become a trend",
    "shifting in a way that feels deliberate",
    "gathering pace in a way that's becoming hard to ignore",
    "accelerating noticeably",
    "moving with a momentum that would be hard to reverse",
    "moving faster than anyone expected",
    "hurtling forward — the speed itself has become a signal",
]

# ---------------------------------------------------------------------------
# Evidence domains — what a citizen might observe
# ---------------------------------------------------------------------------

DOMAINS = [
    {
        "name": "elite_cohesion",
        "weak_cues": [
            "sharp blame-shifting and reversing decisions within hours among leadership",
            "a senior figure who usually keeps a high profile has gone conspicuously quiet",
            "insiders describe meetings ending in shouting, with no resolution",
            "two factions in the inner circle are no longer speaking through intermediaries",
            "a cabinet reshuffle was announced and then reversed within the same day",
            "a once-loyal advisor has been publicly contradicting official statements",
            "key positions have been left unfilled for weeks, suggesting internal deadlock",
            "the inner circle's public appearances have dropped sharply, suggesting withdrawal",
            "a regime insider was overheard saying 'nobody's steering the ship right now'",
            "the leader's closest ally has been conspicuously absent from state functions for over a week",
            "two competing directives were issued on the same day from different ministries, neither withdrawn",
            "a patronage network that kept mid-level officials loyal has visibly stopped delivering",
            "an internal policy document leaked that contradicts the regime's public position entirely",
            "senior figures are giving interviews that subtly distance themselves from the leader's recent decisions",
            "the usual backchannels for resolving elite disputes have gone silent — nobody is mediating",
            "a veteran regime fixer known for quiet problem-solving has been sidelined without explanation",
        ],
        "mixed_cues": [
            "the leadership appears divided on tactics but not on holding power — they disagree on how, not whether",
            "there's visible tension at the top, but both sides seem aware that splitting would be fatal",
            "elite cohesion is strained but not broken — rivalries are being managed rather than resolved",
            "insiders say the top is arguing but still making decisions, just more slowly than before",
            "the leadership is functioning, but with a brittleness that wasn't there six months ago",
            "power is concentrating around a smaller group — whether that's strength or desperation is debatable",
            "elites are publicly unified but privately positioning themselves for multiple outcomes",
            "the inner circle has tightened, but it's unclear if that reflects confidence or paranoia",
        ],
        "strong_cues": [
            "the leadership circle is operating with unusual discipline and coordination",
            "disputes among elites are ending with clear winners and quick public alignment",
            "inner circle figures are making visible shows of unity at public events",
            "senior officials are speaking from the same script with fewer contradictions than usual",
            "the succession of commands is functioning smoothly — no gaps, no improvisation",
            "a former rival of the leader has been publicly praising the regime's direction",
            "elite messaging is synchronized in a way that suggests genuine agreement, not coercion",
            "the inner circle recently expanded to include fresh faces, suggesting confidence in the future",
            "a potential challenger was quietly co-opted with a prestigious appointment — the regime still has carrots to offer",
            "the leader made a rare public appearance with all senior figures present — a deliberate show of solidarity",
            "internal policy debates are being resolved quickly and without leaking — the process is tight",
            "a reshuffled cabinet is functioning with visible efficiency, suggesting the changes were planned, not reactive",
            "elite families are investing in long-term domestic projects — they're not hedging with foreign assets",
            "the regime successfully managed a minor scandal by presenting a unified response within hours",
            "junior officials are competing for promotion rather than looking for exits — the incentive structure holds",
            "a well-placed source says the mood in the presidential palace is 'boring and orderly' — no crises being managed",
        ],
    },
    {
        "name": "security_forces",
        "weak_cues": [
            "security units that normally move as one are showing friction — quiet stalling, paperwork excuses",
            "a commander's orders arrived garbled and were questioned before being carried out",
            "rank-and-file soldiers are grumbling about pay delays and broken promises",
            "an officer your contact trusts said his unit would 'wait for written orders' before acting",
            "reports of soldiers fraternizing with civilians more than usual at checkpoints",
            "a military supply convoy was rerouted without explanation, suggesting logistical confusion",
            "an entire unit failed to report for a scheduled patrol, citing 'equipment issues'",
            "a mid-ranking officer was seen meeting privately with opposition-linked figures",
            "weapons inventory checks have been quietly suspended — someone doesn't want accountability right now",
            "soldiers stationed at a key installation have been rotated out and replaced with untested units",
            "a military court-martial was abruptly suspended, suggesting the chain of command can't enforce discipline",
            "officers who previously competed for operational commands are now requesting desk assignments",
            "an informant in the security services says internal surveillance has shifted from opposition to other security units",
            "two garrison commanders have not responded to a routine readiness assessment — unprecedented in recent memory",
            "military families in the capital have been seen moving belongings to relatives in the provinces",
            "a well-connected retired general privately described the current officer corps as 'demoralized and fracturing'",
        ],
        "mixed_cues": [
            "the military appears professional but cautious — following orders without enthusiasm",
            "security forces are deployed but seem to be avoiding confrontation rather than seeking it",
            "the rank and file seem loyal to their commanders, but it's unclear where the commanders stand",
            "military readiness looks adequate on paper, but the mood in the barracks feels uncertain",
            "the security apparatus is functioning but has shifted to a defensive posture — protecting assets, not projecting power",
            "officers are obeying orders but taking longer to execute them, as if waiting to see what happens next",
            "military patrols continue but their routes have subtly shifted away from politically sensitive areas",
            "the security forces are maintaining presence but seem to be conserving resources rather than demonstrating strength",
        ],
        "strong_cues": [
            "security forces are visibly confident — routine patrols, unhurried posture, no improvisation",
            "military exercises are proceeding on schedule with full participation",
            "a contact in the barracks says morale is solid and orders are flowing cleanly",
            "new equipment and personnel rotations suggest the apparatus is investing, not retreating",
            "security checkpoints have been reduced, suggesting confidence rather than siege mentality",
            "military families are not making unusual preparations — life on base feels normal",
            "a senior officer described the situation as 'well within parameters' without hedging",
            "intelligence services appear to be running routine operations, not crisis-mode surveillance",
            "a new class of officer cadets just graduated with full ceremony — the military is thinking long-term",
            "soldiers at checkpoints are relaxed and joking with civilians — not the posture of a force expecting trouble",
            "the regime recently conducted a large-scale military exercise that went smoothly and on schedule",
            "military pay was delivered early this month — a small but telling signal of organizational confidence",
            "the defense minister gave a lengthy public interview about procurement plans for next year — thinking ahead, not firefighting",
            "a contact says inter-service coordination meetings are happening routinely — no emergency footing",
            "military officers are taking scheduled leave — the system isn't canceling time off or restricting movement",
            "the security apparatus just hired civilian consultants for a modernization project — this is peacetime behavior",
        ],
    },
    {
        "name": "money_and_logistics",
        "weak_cues": [
            "business figures who usually hedge are moving money and families out with coordinated urgency",
            "a well-connected merchant closed his main shop and relocated inventory without explanation",
            "government contracts are being paid late or renegotiated on worse terms",
            "hard currency is getting harder to find through normal channels",
            "several import licenses have been delayed indefinitely, disrupting supply chains",
            "a prominent businessman known for his regime connections has quietly sold off assets",
            "banks are imposing new withdrawal limits, citing 'temporary liquidity management'",
            "fuel prices have spiked at the pump despite no change in official policy",
            "the central bank intervened to defend the currency twice this week — and both times it lost ground",
            "a major contractor who depends on government work has started taking payments in foreign currency only",
            "warehouse space in border towns has become scarce — goods are being positioned for quick export",
            "insurance premiums for commercial properties in the capital have spiked without public explanation",
            "a well-connected accountant says several elite-linked firms are quietly restructuring their debts",
            "the government delayed publishing its monthly budget figures — for the first time in years",
            "informal money changers have raised their spread dramatically, pricing in a devaluation they expect",
            "a major shipment of medical supplies was diverted to a military facility without explanation",
        ],
        "mixed_cues": [
            "the economic situation is strained but not panicked — people are tightening belts, not fleeing",
            "some businesses are pulling back while others are expanding, creating a confusing picture",
            "government payments are arriving but late, and vendors are pricing in the delay",
            "the currency is under pressure but hasn't cracked — traders are nervous but still trading",
            "economic indicators are mixed — retail spending is down but construction permits are up",
            "the business community is hedging rather than fleeing — maintaining operations while reducing exposure",
            "banks are tightening credit standards but still lending — caution, not shutdown",
            "the informal economy is growing, which could mean either declining trust in institutions or normal adaptation",
        ],
        "strong_cues": [
            "business and administrative networks are operating with normal predictability",
            "government payments are arriving on schedule, and contracts are being honored",
            "a banker you know says capital flows look stable — no unusual outflows",
            "foreign investment announcements continue, suggesting outside confidence",
            "new construction projects are being launched, signaling long-term regime confidence",
            "the black-market premium on hard currency has actually narrowed",
            "a major government infrastructure contract was just awarded — business as usual",
            "customs and trade are flowing normally at the borders — no unusual delays or inspections",
            "a new shopping center opened this month with heavy foot traffic — consumer confidence is visible",
            "the government just issued long-term bonds and they were oversubscribed — investors are betting on continuity",
            "small businesses are hiring and expanding — the kind of grassroots confidence that's hard to fake",
            "property prices in the capital remain stable, with no signs of distressed selling",
            "a logistics company just signed a five-year contract with the government — they're not worried about regime change",
            "the central bank's reserves are at their highest level in two years",
            "a well-connected trader described the business climate as 'predictable and profitable' — his highest compliment",
            "agricultural exports are proceeding normally and seasonal patterns are holding — the mundane economy is healthy",
        ],
    },
    {
        "name": "street_mood",
        "weak_cues": [
            "street talk has shifted from complaint to anticipation — people speak as if change is imaginable",
            "jokes about the regime are being told openly in places where that would have been unthinkable",
            "a neighborhood gathering turned into a frank discussion about what comes next",
            "young people are restless and less afraid of being seen talking in groups",
            "a normally cautious shopkeeper said openly that 'something has to give'",
            "graffiti critical of the regime appeared overnight in a central location and hasn't been cleaned",
            "religious leaders are making unusually pointed sermons about justice and change",
            "taxi drivers — usually a reliable barometer — are talking about regime change as realistic",
            "a funeral for a minor public figure turned into an impromptu political gathering without police intervention",
            "women in the market are speaking openly about prices and blaming the government by name — a line they used to avoid",
            "a popular song with barely disguised political lyrics is being played everywhere and nobody is stopping it",
            "children in the street are chanting slogans they clearly learned from adults — the anger has permeated households",
            "a normally apolitical professional class — doctors, teachers, lawyers — has started organizing quietly",
            "neighborhood watch groups have subtly shifted from crime prevention to political discussion forums",
            "the mood at a local tea house has changed from grumbling to something more purposeful — people are comparing notes",
            "a well-attended community meeting ended with a spontaneous moment of silence 'for what's coming' — and everyone understood",
        ],
        "mixed_cues": [
            "people are frustrated but fatalistic — complaining more, but not in a way that suggests action",
            "the public mood is angry but fragmented — everyone hates the regime but nobody trusts anyone else",
            "there's a sense of waiting — not satisfaction, not mobilization, just suspension",
            "conversations about politics are more common but still careful, as if testing boundaries",
            "some neighborhoods feel electric with potential while others are completely passive — the mood is uneven",
            "people express anger in private but conformity in public — the gap between the two is growing",
            "younger people are visibly more restless than older people, who counsel patience and caution",
            "there's dark humor everywhere — people cope through jokes, but the jokes have an edge they didn't have before",
        ],
        "strong_cues": [
            "public frustration reads as exhausted rather than mobilizing — cynicism without momentum",
            "people complain but do it like people who expect tomorrow to look much like today",
            "gatherings disperse quickly and conversation turns careful when outsiders approach",
            "a neighbor who used to be outspoken has gone quiet, saying it's 'not worth the trouble'",
            "market vendors are going about their business with the resigned calm of people who've seen worse",
            "public events and festivals are proceeding normally, with ordinary turnout",
            "a community elder said people are 'tired of hoping for change' and just want stability",
            "even the usual troublemakers seem subdued — the energy for protest feels depleted",
            "families are focused on daily survival — school fees, food prices, medical bills — politics feels abstract",
            "a teacher described her students as 'completely uninterested in politics' — a generational shift toward disengagement",
            "people are investing in home improvements and small businesses — behavior that signals expectation of continuity",
            "a wedding celebration proceeded for three days without any political discussion — normalcy is dominant",
            "the local football league is drawing big crowds — people have leisure time and are using it for entertainment, not protest",
            "a normally restless neighborhood feels calm and settled — children playing, shops busy, nothing unusual",
            "street vendors have returned to areas they'd avoided during previous periods of tension — the all-clear has been internalized",
            "a veteran community organizer said 'this isn't the moment' and went back to his regular work — the reading from the ground is stability",
        ],
    },
    {
        "name": "information_control",
        "weak_cues": [
            "messaging from the top is forceful but oddly repetitive, aimed at reassuring insiders not the public",
            "the state broadcaster went off-air for several hours with no explanation",
            "censorship is getting sloppier — critical posts survive longer before deletion",
            "an official denial was issued for something nobody had publicly accused them of",
            "state media contradicted itself twice in the same broadcast",
            "a government spokesperson appeared visibly flustered when asked a routine question",
            "leaked documents are circulating faster than the regime can discredit them",
            "internet speeds have been throttled in the capital without official acknowledgment",
            "a pro-regime commentator went off-script on live television and wasn't cut off — the control room is distracted",
            "underground pamphlets are being distributed in neighborhoods that were previously too surveilled for such activity",
            "the regime's social media operation has become visibly disorganized — bot accounts are contradicting each other",
            "a state-owned newspaper printed a retraction of its own editorial from the previous day — editorial control is slipping",
            "foreign media crews have been operating with unusual freedom in the capital — the usual minders are absent",
            "a well-known journalist who was previously silenced has resumed publishing critical pieces without consequence",
            "the regime's daily press briefing has been cancelled three times this week — they can't decide what to say",
            "VPN usage has spiked dramatically and the regime hasn't responded — either they can't detect it or can't be bothered",
        ],
        "mixed_cues": [
            "the regime's messaging is active but slightly off-key — trying harder than usual to project normalcy",
            "censorship is functional but reactive rather than proactive — they're playing catch-up",
            "state media sounds confident on camera, but insiders say the editorial line changes daily",
            "information control seems to be holding, but at increasing cost in resources and credibility",
            "the regime is still shaping the narrative but has lost the initiative — they're responding to events rather than framing them",
            "media coverage is controlled but subtly less enthusiastic — the tone has shifted from celebratory to merely functional",
            "some critical content is getting through, but it's unclear if that's tolerance, incompetence, or a deliberate pressure valve",
            "the regime's information apparatus looks intact but tired — producing content without the energy or creativity it once had",
        ],
        "strong_cues": [
            "the regime's messaging is coordinated across channels with fewer internal contradictions",
            "information control is tight and efficient — leaks are rare and quickly contained",
            "state media is running confident, forward-looking programming, not defensive messaging",
            "critics are being handled through legal channels rather than crude suppression",
            "the regime just launched a new public communications initiative — a sign of long-term planning",
            "foreign journalists report unusual but professional cooperation from government press offices",
            "social media monitoring appears sophisticated and well-resourced",
            "opposition voices online are being countered with detailed rebuttals, not just deleted",
            "the regime unveiled a glossy new state media platform — they're investing in propaganda, not just maintaining it",
            "a pro-regime influencer campaign is running smoothly and gaining traction with young people",
            "the government held an open press conference and handled hostile questions with visible confidence",
            "a potentially damaging story was contained within hours through a coordinated counter-narrative — impressive response time",
            "independent media outlets are operating but finding it difficult to gain audience — the regime's narrative dominance is structural",
            "the government's public data portal was updated on schedule with detailed statistics — transparency as a strength signal",
            "an opposition figure's leaked criticism was preemptively addressed before it gained traction — the intelligence apparatus is ahead of events",
            "state media is running human-interest stories about development projects — the messaging is confident enough to be boring",
        ],
    },
    {
        "name": "personal_observations",
        "weak_cues": [
            "your superior at work has been evasive and cancelled two meetings without rescheduling",
            "a reliable contact who usually knows everything said 'I honestly don't know what's happening'",
            "people you trust are making quiet preparations — nothing dramatic, but noticeable",
            "a relative in the civil service mentioned that personnel transfers have frozen unexpectedly",
            "your neighbor, who works in a ministry, has started keeping a packed bag by the door",
            "a colleague who never discusses politics asked you hypothetically what you'd do 'if things changed'",
            "three different people you trust have independently suggested you 'be careful' this month",
            "the local clinic has been quietly stocking extra supplies, according to a nurse you know",
            "your landlord — a regime supporter — quietly asked whether you had relatives abroad, 'just in case'",
            "a friend who works in banking mentioned that several high-profile clients have moved money offshore this week",
            "your cousin in the army hasn't called in two weeks — unusual for someone who calls every Sunday",
            "a parent at your child's school who works in intelligence seemed distracted and left early, looking worried",
            "your local pharmacist has been hoarding antibiotics and painkillers — not for sale, for personal stock",
            "a colleague returned from a business trip and said foreign contacts were asking unusually pointed questions about stability",
            "someone you barely know stopped you on the street and asked quietly 'have you heard anything?'",
            "your phone has been making unusual clicking sounds — probably nothing, but in this context it feels ominous",
        ],
        "mixed_cues": [
            "people seem watchful rather than panicked — paying closer attention but not acting differently",
            "your contacts are divided — some see trouble coming, others insist nothing has really changed",
            "life feels normal on the surface but there's an undercurrent of alertness that wasn't there before",
            "a well-connected friend said 'it could go either way' and seemed genuinely unsure",
            "some of your friends are making contingency plans while others are booking vacations — the signals conflict",
            "your neighborhood is quieter than usual, but you can't tell if that's tension or just a quiet week",
            "people are paying attention to the news more carefully, but nobody's changing their daily routines yet",
            "a friend who's usually pessimistic and a friend who's usually optimistic gave you the exact same assessment: 'uncertain'",
        ],
        "strong_cues": [
            "contacts who tend to be alarmist have gone strangely calm, describing things as 'ugly but stable'",
            "your workplace is functioning normally — no unusual absences or anxious corridor talk",
            "a well-connected friend said the situation is 'messy but controlled' and seemed unbothered",
            "routine government services are running without the delays that usually signal internal chaos",
            "your children's school is operating normally, and other parents seem unconcerned",
            "a neighbor who works in security mentioned casually that 'nothing unusual' is happening",
            "people who were nervous a month ago have visibly relaxed — the tension has dissipated",
            "a civil servant friend described the mood at work as 'boring' — which in this context is reassuring",
            "your local market is well-stocked and prices haven't spiked — supply chains are intact",
            "a friend in the medical profession says hospital admissions are routine — no surge in stress-related illness",
            "your relatives in the provinces report completely normal conditions — the anxiety seems limited to political circles",
            "a neighbor just started a home renovation project — you don't invest in your house if you think you might need to leave",
            "friends are making plans for holidays months from now — they're not thinking in terms of crisis timelines",
            "a retired military contact described the current situation as 'one of the calmer periods I can remember'",
            "your children came home from school with no unusual stories — if teachers were worried, it would show in the classroom",
            "a well-informed friend offered to lend you a book 'because there's nothing happening worth worrying about'",
        ],
    },
    {
        "name": "diplomatic_signals",
        "weak_cues": [
            "foreign embassies have quietly updated their travel advisories with unusually specific warnings",
            "a diplomat at a social function asked pointed questions about 'contingency thinking'",
            "international organizations have begun relocating non-essential staff, citing 'precaution'",
            "a friendly foreign government issued a statement calling for 'dialogue' — diplomatic code for concern",
            "visa processing at several embassies has slowed dramatically, suggesting internal reassessment",
            "foreign correspondents who had left are returning — they smell a story",
            "a foreign military attaché was spotted meeting with opposition figures at a hotel — either poorly concealed or deliberately visible",
            "an international NGO that normally coordinates with the government has begun operating independently",
            "foreign airlines have quietly added extra flights out of the capital — route changes driven by demand, not schedule",
            "a neighboring country has reinforced its border patrols — they're preparing for a refugee scenario",
            "the UN country representative requested a private meeting with senior officials — unusual for routine operations",
            "foreign business delegations have cancelled upcoming visits, citing 'scheduling conflicts' that nobody believes",
            "diplomatic sources say several embassies have activated their crisis management protocols",
            "a friendly foreign intelligence contact said 'we're watching very carefully' — which is diplomat for alarm",
            "international shipping companies have reclassified the country as higher-risk for cargo insurance",
            "the regime's ambassador to a key ally was recalled for 'consultations' — a move that suggests the ally is distancing",
        ],
        "mixed_cues": [
            "foreign governments are watching closely but haven't changed their posture — monitoring, not reacting",
            "diplomatic contacts describe the situation as 'fluid' — the word diplomats use when they genuinely don't know",
            "international coverage has increased but remains balanced — crisis is possible, not certain",
            "foreign embassies are maintaining full operations but have updated contingency plans — standard prudence or genuine concern",
            "a regional power issued a carefully worded statement supporting 'institutional stability' — ambiguous enough to cover any outcome",
            "international organizations are continuing their programs but have shortened planning horizons from annual to quarterly",
            "foreign analysts are split — some see a crisis developing, others see managed turbulence within normal bounds",
            "the diplomatic community is having more frequent informal gatherings — sharing information, not yet coordinating action",
        ],
        "strong_cues": [
            "foreign embassies are operating normally and have not changed their advisory levels",
            "international investors just signed a major deal with the government — a vote of confidence",
            "a senior diplomat described the regime as 'firmly in control' at a recent reception",
            "foreign military attachés report no unusual activity — routine operations only",
            "international rating agencies have maintained their assessment without comment",
            "the regime just secured a new bilateral agreement, suggesting external partners see stability",
            "a major international conference is scheduled to be hosted in the capital next quarter — foreign participation is confirmed",
            "foreign development agencies just approved a multi-year funding package — they don't commit funds to countries they expect to collapse",
            "diplomatic social events are proceeding with normal attendance — no embassies are in bunker mode",
            "a regional security organization invited the country to join a new cooperation framework — a signal of normalized relations",
            "foreign journalists based in the country describe their working conditions as 'routine' — no unusual restrictions",
            "international trade delegations continue to arrive on schedule — commercial confidence is intact",
            "a neighboring country's leader made a friendly state visit this week — you don't visit allies you think are about to fall",
            "the World Bank just praised the country's recent fiscal management — institutional approval is a stability signal",
            "foreign students continue to enroll at local universities — their embassies haven't advised them to leave",
            "the country's sovereign credit spread has tightened slightly — financial markets are pricing in stability, not risk",
        ],
    },
    {
        "name": "institutional_functioning",
        "weak_cues": [
            "court proceedings have been suspended or delayed without explanation in several jurisdictions",
            "civil servants are reporting contradictory instructions from different parts of the bureaucracy",
            "a scheduled legislative session was postponed at the last minute — officially for 'procedural reasons'",
            "government offices that usually run like clockwork are experiencing unexplained staffing gaps",
            "municipal services — garbage collection, water, permits — have become erratic in the capital",
            "a routine government audit was abruptly cancelled, raising questions about what it might have found",
            "the national statistics office delayed a scheduled data release without explanation — they're hiding something or can't agree on what to publish",
            "a regulatory agency issued contradictory rulings on the same case within a week",
            "public hospitals are reporting supply shortages for items that were available last month — the procurement system is breaking down",
            "teachers at government schools haven't been paid in six weeks and are staging informal slowdowns",
            "the government's online services portal has been offline for days — a small thing, but symptomatic",
            "a scheduled census operation was indefinitely postponed — the state can't organize basic logistics right now",
            "building permits that normally take a week are taking months, with no one willing to sign off",
            "the postal service has effectively stopped delivering to several districts — institutional reach is contracting",
            "a government database was breached and the response was confused and delayed — IT infrastructure is neglected",
            "emergency services responded to a recent incident with visible disorganization — the coordination mechanisms are fraying",
        ],
        "mixed_cues": [
            "institutions are functioning but with visible strain — delays are longer, tempers are shorter",
            "the bureaucracy is still processing paperwork, but decisions that used to take days now take weeks",
            "government services work in some areas and not others — the pattern suggests dysfunction, not collapse",
            "the system is creaking but not collapsing — people are working around the bottlenecks rather than through them",
            "some institutions are functioning better than others — the unevenness itself is telling",
            "routine government business continues but with more errors and corrections than usual",
            "the legal system is processing cases but with unexplained scheduling irregularities",
            "public services are maintained but at reduced quality — the baseline has shifted downward without anyone announcing it",
        ],
        "strong_cues": [
            "government institutions are processing routine business without unusual delays",
            "the courts are functioning normally, including cases that touch on politically sensitive issues",
            "civil service recruitment is proceeding — the bureaucracy is hiring, not freezing",
            "a scheduled policy announcement was delivered on time with a detailed implementation plan",
            "municipal services are running smoothly — a mundane but reliable indicator of institutional health",
            "tax collection is proceeding normally, suggesting the fiscal apparatus remains intact",
            "a new government regulation was drafted, debated, and implemented through normal channels — the policy process works",
            "the education ministry published its annual report on time with detailed performance metrics",
            "public transportation is running on schedule and recently expanded a route — investment in routine services continues",
            "a freedom-of-information request was processed within the statutory timeframe — rule of law is being observed",
            "the national health system successfully managed a minor disease outbreak through standard protocols — institutional capacity is real",
            "government IT systems were recently upgraded — someone is thinking about long-term infrastructure",
            "a disputed land case was resolved through the courts without political interference — the judiciary is independent enough to function",
            "environmental inspections are being conducted on schedule — even the less glamorous parts of government are operational",
            "the civil service pension fund made its payments on time — the state is honoring its long-term obligations",
            "a new cohort of civil servants completed their training program and were assigned to posts — the system is reproducing itself normally",
        ],
    },
]

# ---------------------------------------------------------------------------
# Coordination climate phrases
# ---------------------------------------------------------------------------

COORDINATION_QUIET = [
    "People are careful. Conversations happen in whispers, behind closed doors, in coded language.",
    "Nobody is saying anything openly. If people are thinking about action, they're keeping it invisible.",
    "The silence is total — not peaceful silence, but the kind where everyone is watching everyone else.",
    "If there's any coordination happening, it's invisible to you — and you're paying attention.",
    "Even trusted friends change the subject when talk gets too concrete. Fear is doing its work.",
    "People keep their thoughts locked away. The cost of being overheard is too high to risk candor.",
    "Informants are everywhere, or at least everyone acts as though they are. Self-censorship is total.",
    "The atmosphere is one of enforced normalcy — people going through motions with their real thoughts hidden.",
    "You haven't had a single honest political conversation in weeks, and you know others haven't either.",
    "The regime's surveillance — real or imagined — has atomized everyone into private isolation.",
]

COORDINATION_GUARDED = [
    "People are careful, but the silence has cracks. Close friends will say things they wouldn't say to anyone else.",
    "There's a layer of deniable communication — people share news articles or jokes that carry a political subtext.",
    "Trust networks are small and closed. People talk within families and old friendships, nowhere else.",
    "You notice people choosing their words more carefully than usual, not out of fear but out of caution about who's listening.",
    "Information circulates in whisper networks — you hear things third-hand, never from the original source.",
    "People are aware of each other's frustrations but nobody is naming them directly. It's communication through omission.",
    "Small gestures of solidarity happen — a knowing look, a shared silence — but nothing that could be called coordination.",
    "The mood is guarded rather than silent. People want to talk but don't yet trust that talking is safe.",
]

COORDINATION_TENTATIVE = [
    "There are tentative feelers — indirect questions, hypothetical conversations that feel less hypothetical.",
    "People are testing the waters with careful ambiguity, saying things that could be taken two ways.",
    "You sense that others are thinking what you're thinking, but nobody wants to be the first to say it plainly.",
    "Conversations are happening, but in that deniable way — 'what if' rather than 'when'.",
    "A few people have dropped hints that they'd act if others did. No one has committed to anything.",
    "The mood is one of cautious mutual recognition — people suspect they're not alone, but can't be sure.",
    "Some are starting to speak a little more freely, but always with an exit — 'I'm just thinking out loud.'",
    "There's a careful economy of glances and half-sentences — people communicating without quite speaking.",
    "Small groups are forming around shared frustrations, but they dissolve quickly if someone unfamiliar approaches.",
    "People are sharing political memes and coded messages on social media — deniable but clearly intentional.",
]

COORDINATION_MOBILIZING = [
    "The conversation has shifted from 'should we' to 'how would we' — still speculative, but with operational undertones.",
    "People who were cautious a week ago are now actively reaching out to expand their networks.",
    "There's a palpable sense that a critical mass is forming, even if nobody can count the numbers yet.",
    "Trusted intermediaries are moving between groups, relaying assessments and gauging willingness.",
    "The fear of speaking hasn't disappeared, but it's being overridden by a sense that silence is now the riskier choice.",
    "People are starting to make commitments — small ones, conditional ones, but commitments nonetheless.",
    "The coordination is still informal, but it has a direction and momentum that feels qualitatively different from venting.",
    "Networks that were dormant are reactivating — people who haven't spoken in months are suddenly in contact.",
]

COORDINATION_OPEN = [
    "People are speaking as if coordination is already happening — 'when it happens' rather than 'if'.",
    "Conversations have shifted from whispers to open talk. People are less afraid of being overheard.",
    "The mood has crossed a line — people are acting like the question isn't whether to act, but when.",
    "Groups are forming openly, and the old caution about being seen together has largely dissolved.",
    "People are making plans, not just venting. The conversations have a logistical quality now.",
    "The sense of isolation has broken — people realize they're not alone, and they're acting on it.",
    "Coordination is happening in plain sight — meetings, signals, preparations that would have been unthinkable a month ago.",
    "The fear barrier has collapsed. People who were terrified last week are now openly discussing action.",
    "There's an almost festive quality to the defiance — the secrecy has dropped away and people feel liberated by it.",
    "Strangers are approaching each other to talk about 'the situation' — the trust radius has expanded dramatically.",
]

# ---------------------------------------------------------------------------
# Bottom-line templates
# ---------------------------------------------------------------------------

BOTTOMLINE_VERY_WEAK = [
    "The regime is in serious trouble. The signs are converging and the trajectory is unmistakable.",
    "This is not erosion — this is structural failure in progress. Multiple systems are failing simultaneously.",
    "The regime's ability to project power has degraded to the point where its survival is an open question.",
    "Everything points in the same direction: the regime is weaker than it has been in living memory.",
    "The institutional foundations that held this regime together are visibly crumbling.",
    "If you were looking for a textbook picture of a regime approaching collapse, this is close to what you'd expect.",
]

BOTTOMLINE_WEAK = [
    "The regime looks brittle in ways that are becoming harder to conceal.",
    "Signs of institutional erosion are accumulating faster than the regime can manage them.",
    "The regime's position has weakened meaningfully, and the trajectory is not stabilizing.",
    "Multiple pillars of regime support are showing cracks simultaneously — this is not business as usual.",
    "The regime is losing the initiative, reacting to events rather than shaping them.",
    "What you're seeing is not a rough patch — it's a pattern of accelerating deterioration.",
]

BOTTOMLINE_WEAK_MODERATE = [
    "The regime is under pressure, but whether this pressure is fatal or manageable remains genuinely open.",
    "There are real signs of strain — not yet crisis-level, but trending in a direction that should concern the regime.",
    "The regime has lost some of its aura of invincibility, though it retains significant capacity.",
    "Things are slipping, but slowly. The regime still has cards to play — the question is whether it will play them in time.",
    "Cracks are visible to anyone paying attention, but the structure hasn't buckled. Yet.",
]

BOTTOMLINE_WEAK_BORDERLINE = [
    "The regime is strained in ways that lean toward vulnerability, but the picture isn't decisive.",
    "More signals point toward weakness than strength, but the margin is thin and could shift.",
    "The weight of evidence tilts toward a regime that's losing ground, though it hasn't lost control.",
    "There's a pattern forming that looks like decline, but it's early enough that recovery remains possible.",
    "If forced to call it, you'd lean toward trouble — but you'd hold that assessment lightly.",
    "The regime's position is deteriorating, but at a pace that still allows for course correction if they act.",
]

BOTTOMLINE_BORDERLINE = [
    "The regime's grip shows strain, but the picture is inconsistent enough that a decisive read feels premature.",
    "The situation could tip either way — there are credible signals pointing in both directions.",
    "Something is shifting, but whether it represents real vulnerability or managed turbulence is genuinely unclear.",
    "For every sign of weakness, there's a corresponding sign of resilience — the net picture is ambiguous.",
    "The honest assessment is uncertainty. The regime is neither collapsing nor thriving.",
    "You can build a case for either outcome using the same evidence — that's how mixed the signals are.",
    "The situation is balanced on a knife's edge — small events could determine which way it falls.",
]

BOTTOMLINE_STRONG_BORDERLINE = [
    "The balance of signals leans slightly toward stability, but not enough to be confident.",
    "More things are working than not, but the margin of comfort is narrow.",
    "The regime appears to be managing its challenges, though the effort required seems to be growing.",
    "If you had to bet, you'd bet on continuity — but you wouldn't bet much.",
    "The situation reads as stable-for-now, with enough underlying stress that the 'for now' matters.",
    "The regime has more going for it than against it, but the positive indicators aren't overwhelming.",
]

BOTTOMLINE_STRONG_MODERATE = [
    "The regime is under some pressure but appears to be managing it — stressed but not endangered.",
    "Despite surface-level turbulence, the fundamental structures of control seem to be holding.",
    "The regime has problems, but they're the kind of problems regimes survive, not the kind that bring them down.",
    "There's noise, but the signal underneath suggests continuity rather than rupture.",
    "The balance of evidence leans toward stability, though not so decisively that surprises are impossible.",
]

BOTTOMLINE_STRONG = [
    "The regime appears steadier than the rumors suggest, and the system is behaving like it expects to endure.",
    "Current indicators point to maintained control, with internal challenges being managed rather than spiraling.",
    "The regime's position is more consolidated than surface-level noise would suggest.",
    "The system is operating with a confidence that would be hard to fake — this looks like genuine stability.",
    "Opposition to the regime exists, but it lacks the critical mass, coordination, or timing to pose a real threat.",
    "By every measure available to you, the regime's hold on power is firm and likely to remain so.",
]

BOTTOMLINE_VERY_STRONG = [
    "The regime is as secure as any authoritarian system can realistically be. All indicators point to consolidated power.",
    "This regime is not just surviving — it's thriving. The opposition is fragmented, the apparatus is loyal, and the population is quiescent.",
    "By every available metric, the regime's hold on power is overwhelming and shows no signs of weakening.",
    "The system is operating with a degree of confidence and institutional coherence that leaves almost no room for a successful challenge.",
    "Any attempt to move against this regime would face extraordinary odds — the structures of control are deep and functional.",
    "The regime's position is as strong as you've seen it, and the conditions for challenge are as unfavorable as they've been in years.",
]

# ---------------------------------------------------------------------------
# "What's unclear" templates
# ---------------------------------------------------------------------------

UNCLEAR_WEAK = [
    "Whether the top is losing control, or intentionally tightening the circle and freezing everyone else out.",
    "How much of the disarray is structural versus a temporary response to a specific crisis.",
    "Whether the military's silence means they're loyal, undecided, or waiting for the right moment.",
    "If the economic deterioration is perceived by ordinary people as the regime's fault or as bad luck.",
    "Whether any of the apparent weakness is being deliberately staged to flush out opponents.",
    "How deep the fractures go — surface-level dysfunction can mask either deeper rot or functioning parallel systems.",
    "Whether the security apparatus would follow orders to suppress a large-scale mobilization, or stand aside.",
    "If the regime has foreign backers who might intervene to prop it up at the last moment.",
]

UNCLEAR_WEAK_BORDERLINE = [
    "Whether the strain you're seeing is the beginning of the end, or just a rough patch the regime will weather.",
    "How much of the opposition's apparent confidence is based on real intelligence versus wishful thinking.",
    "Whether the regime's recent mistakes reflect incompetence or distraction from a more serious internal struggle.",
    "If the regime still has the capacity to surprise — to pull a rabbit out of the hat when it matters most.",
    "Whether the economic pressure is reaching the people who matter to the regime, or only those who don't.",
    "How much of what you're hearing is signal versus noise — in uncertain times, rumors multiply faster than facts.",
    "Whether the regime's reduced visibility is a sign of weakness or a deliberate retreat into fortified positions.",
    "If there's a faction within the regime that might cut a deal to preserve itself at the expense of the leadership.",
]

UNCLEAR_BORDERLINE = [
    "The same facts can be read as cracking foundations or as a controlled consolidation — the evidence genuinely supports both.",
    "Whether the opposition is actually coordinating, or just imagining coordination into existence.",
    "Whether the regime's apparent calm is genuine confidence or a carefully maintained facade.",
    "How much the outside world's assessment matters — external actors may be better or worse informed than you.",
    "Whether the quiet you're seeing is the quiet before a storm or the quiet of a situation settling down.",
    "Whether small acts of defiance signal a broader shift or are isolated incidents that will peter out.",
    "How much weight to give to your own assessment versus the assessments of people you trust who disagree with you.",
    "Whether the timing of recent events is coincidental or coordinated — the pattern could go either way.",
]

UNCLEAR_STRONG_BORDERLINE = [
    "Whether the stability you're seeing is structural or just a temporary equilibrium that could shift quickly.",
    "If there are hidden vulnerabilities that the regime's confident exterior is designed to conceal.",
    "Whether the opposition has genuinely been defeated or has simply gone underground to regroup.",
    "How much of the current calm depends on specific individuals whose removal could change everything.",
    "Whether the regime's recent successes have made it complacent about emerging threats it should be managing.",
    "If external conditions change — an economic shock, a regional crisis — whether the regime's stability would hold.",
    "Whether the population's quiescence reflects genuine acceptance or merely the absence of a triggering event.",
    "How much of the regime's apparent strength depends on the weakness and disorganization of the opposition, rather than its own merit.",
]

UNCLEAR_STRONG = [
    "The regime could still be masking trouble behind a disciplined exterior, but current indicators don't support that reading.",
    "If unrest flares, the response capacity seems intact — though no system is invulnerable to surprises.",
    "Whether any latent opposition exists that simply hasn't shown itself yet — absence of evidence is not evidence of absence.",
    "How long the current stability can hold if underlying economic or social pressures continue to build.",
    "Whether the regime's strength is being accurately perceived by potential challengers — overestimation could be self-fulfilling.",
    "If the regime's information advantage is as complete as it appears, or if opponents are communicating through channels you can't see.",
    "Whether the current generation of potential challengers has the will or capacity to act even if circumstances change.",
    "How much of the regime's stability depends on continued economic performance that may not be sustainable indefinitely.",
]


# ---------------------------------------------------------------------------
# Slider computations
# ---------------------------------------------------------------------------

def _compute_sliders(z_score, cutoff_center=0.0, clarity_width=1.0,
                     direction_slope=0.8, coordination_slope=0.6):
    """Compute three latent sliders from a z-score.

    Parameters
    ----------
    z_score : float
        (x_i - z) / sigma. Negative = regime seems weak.
    cutoff_center : float
        Where the theoretical cutoff sits in z-score space.
    clarity_width : float
        Width of the ambiguous region around the cutoff.
    direction_slope : float
        Steepness of the direction logistic. Lower = more gradual transition.
    coordination_slope : float
        Steepness of the coordination logistic.

    Returns
    -------
    direction : float in [0, 1]
        0 = very weak, 1 = very strong. Monotone in z_score.
    clarity : float in [0, 1]
        0 = maximally ambiguous, 1 = maximally clear.
        U-shaped: lowest near cutoff_center.
    coordination : float in [0, 1]
        0 = totally quiet, 1 = openly discussing action.
        Monotone decreasing in z_score (weak regime → more open talk).
    """
    centered = z_score - cutoff_center

    direction = 1.0 / (1.0 + np.exp(-centered * direction_slope))

    distance_from_cutoff = abs(centered)
    clarity = 1.0 - np.exp(-(distance_from_cutoff / clarity_width) ** 2)

    coordination = 1.0 / (1.0 + np.exp(centered * coordination_slope))

    return direction, clarity, coordination


def _pick_rung(ladder, value, rng, rung_bias=0.0):
    """Pick a rung from a phrase ladder based on a [0,1] value.

    Adds slight jitter so nearby values can land on different rungs.
    rung_bias shifts the index (positive = higher/more intense rungs).
    """
    n = len(ladder)
    # Map value to index with small noise
    idx = value * (n - 1) + rng.normal(0, 0.4) + rung_bias
    idx = int(np.clip(round(idx), 0, n - 1))
    return ladder[idx]


def _sample_evidence_item(domain, direction, clarity, rng,
                          dissent_floor=0.25, mixed_cue_clarity=0.5,
                          rung_bias=0.0):
    """Sample one evidence bullet from a domain.

    Parameters
    ----------
    domain : dict
        One of the DOMAINS entries.
    direction : float
        0 = weak regime, 1 = strong regime.
    clarity : float
        0 = ambiguous, 1 = clear.
    dissent_floor : float
        Minimum probability of a contrary-valence cue. 0.25 means even at
        extreme signals, 25% of cues will contradict the dominant direction.
        Ensures every briefing has meaningful dissent.
    mixed_cue_clarity : float
        Clarity threshold below which mixed (ambiguous) cues can appear.

    Returns
    -------
    bullet : str
        A single evidence observation.
    """
    # Valence: probability of picking a "destabilizing" (weak) cue
    # Compressed to [dissent_floor, 1 - dissent_floor] so there's always dissent
    valence_range = 1.0 - 2.0 * dissent_floor
    p_destabilizing = dissent_floor + valence_range * (1.0 - direction)

    # Near the cutoff (low clarity), use mixed cues from the domain
    if clarity < mixed_cue_clarity and "mixed_cues" in domain:
        p_mixed = 1.0 - clarity / mixed_cue_clarity  # 1.0 at clarity=0, 0.0 at threshold
        if rng.random() < p_mixed:
            cue = rng.choice(domain["mixed_cues"])
            corroboration = _pick_rung(CORROBORATION_LADDER, dissent_floor + clarity, rng, rung_bias=rung_bias)
            if rng.random() < 0.5:
                return f"{cue} ({corroboration})"
            return cue

    use_weak = rng.random() < p_destabilizing

    if use_weak:
        cue = rng.choice(domain["weak_cues"])
    else:
        cue = rng.choice(domain["strong_cues"])

    # Intensity and corroboration depend on clarity
    intensity = _pick_rung(INTENSITY_LADDER, direction if not use_weak else (1 - direction), rng, rung_bias=rung_bias)
    corroboration = _pick_rung(CORROBORATION_LADDER, clarity, rng, rung_bias=rung_bias)

    # Compose: cue + framing elements (vary the structure)
    roll = rng.random()
    if roll < 0.4:
        return f"{cue} ({corroboration})"
    elif roll < 0.6:
        modality = _pick_rung(MODALITY_LADDER, clarity, rng, rung_bias=rung_bias)
        return f"This {modality} {intensity} — {cue}"
    elif roll < 0.8:
        momentum = _pick_rung(MOMENTUM_LADDER, direction, rng, rung_bias=rung_bias)
        return f"{cue} — the trend {momentum}"
    else:
        return cue


def _seed_for_agent_period(base_seed, agent_id: int, period: int) -> int:
    """Stable per-agent-period seed to keep briefings reproducible but distinct."""
    return abs(deterministic_hash((base_seed or 0, agent_id, period))) % (2**31)


def _validate_cutpoints(name: str, cuts, expected_len: int, min_gap: float = 1e-6) -> tuple[float, ...]:
    """Validate monotone cutpoints in (0,1) and return a canonical tuple."""
    if cuts is None:
        raise ValueError(f"{name} cannot be None")

    values = tuple(float(x) for x in cuts)
    if len(values) != expected_len:
        raise ValueError(f"{name} must have length {expected_len}; got {len(values)}")
    if not all(0.0 < v < 1.0 for v in values):
        raise ValueError(f"{name} entries must be in (0,1): {values}")
    for a, b in zip(values, values[1:]):
        if b - a < min_gap:
            raise ValueError(f"{name} must be strictly increasing: {values}")
    return values


def _validate_language_variant(language_variant: str) -> str:
    value = str(language_variant or DEFAULT_LANGUAGE_VARIANT).strip().lower()
    if value not in SUPPORTED_LANGUAGE_VARIANTS:
        raise ValueError(
            f"language_variant must be one of {SUPPORTED_LANGUAGE_VARIANTS}; got {language_variant!r}"
        )
    return value


def normalize_language_variant(language_variant: str) -> str:
    """Return canonical language variant name (accepts aliases)."""
    return _validate_language_variant(language_variant)


def _format_domain_name(domain_name: str) -> str:
    return str(domain_name).replace("_", " ").strip().title()


def _clarity_label(clarity: float) -> str:
    if clarity < 0.2:
        return "very ambiguous"
    if clarity < 0.4:
        return "ambiguous"
    if clarity < 0.65:
        return "mixed"
    if clarity < 0.85:
        return "fairly clear"
    return "clear"


def _pick_bottom_line(direction: float, rng, cuts=DEFAULT_BOTTOMLINE_CUTS) -> str:
    """Map direction slider to one bottom-line sentence."""
    c1, c2, c3, c4, c5, c6, c7, c8 = cuts
    if direction < c1:
        return rng.choice(BOTTOMLINE_VERY_WEAK)
    if direction < c2:
        return rng.choice(BOTTOMLINE_WEAK)
    if direction < c3:
        return rng.choice(BOTTOMLINE_WEAK_MODERATE)
    if direction < c4:
        return rng.choice(BOTTOMLINE_WEAK_BORDERLINE)
    if direction < c5:
        return rng.choice(BOTTOMLINE_BORDERLINE)
    if direction < c6:
        return rng.choice(BOTTOMLINE_STRONG_BORDERLINE)
    if direction < c7:
        return rng.choice(BOTTOMLINE_STRONG_MODERATE)
    if direction < c8:
        return rng.choice(BOTTOMLINE_STRONG)
    return rng.choice(BOTTOMLINE_VERY_STRONG)


def _pick_unclear_items(direction: float, rng, n_items: int = 2, cuts=DEFAULT_UNCLEAR_CUTS) -> list[str]:
    """Sample uncertainty bullets from the direction-appropriate uncertainty pool."""
    c1, c2, c3, c4, c5 = cuts
    if direction < c1:
        unclear_pool = UNCLEAR_WEAK
    elif direction < c2:
        unclear_pool = UNCLEAR_WEAK_BORDERLINE
    elif direction < c3:
        unclear_pool = UNCLEAR_BORDERLINE
    elif direction < c4:
        unclear_pool = UNCLEAR_STRONG_BORDERLINE
    elif direction < c5:
        unclear_pool = UNCLEAR_STRONG
    else:
        unclear_pool = UNCLEAR_STRONG
    return list(rng.choice(unclear_pool, size=min(n_items, len(unclear_pool)), replace=False))


def _pick_atmosphere(
    coordination: float,
    rng,
    cuts=DEFAULT_COORDINATION_CUTS,
    blend_prob: float = DEFAULT_COORDINATION_BLEND_PROB,
) -> str:
    """Map coordination slider to one atmosphere sentence.

    5 levels: quiet → guarded → tentative → mobilizing → open
    5 cutpoints partition [0,1] into 6 regions; the two blend zones
    sit between adjacent levels.
    """
    c1, c2, c3, c4, c5 = cuts
    if coordination > c5:
        return rng.choice(COORDINATION_OPEN)
    if coordination > c4:
        # Blend mobilizing and open
        if rng.random() < blend_prob:
            return rng.choice(COORDINATION_MOBILIZING)
        return rng.choice(COORDINATION_OPEN)
    if coordination > c3:
        return rng.choice(COORDINATION_MOBILIZING)
    if coordination > c2:
        return rng.choice(COORDINATION_TENTATIVE)
    if coordination > c1:
        return rng.choice(COORDINATION_GUARDED)
    return rng.choice(COORDINATION_QUIET)


@dataclass
class Briefing:
    """A structured intelligence briefing for one agent."""
    bottom_line: str
    observations: list[str]
    unclear: list[str]
    atmosphere: str
    # Hidden metadata for analysis
    z_score: float
    direction: float
    clarity: float
    coordination: float
    language_variant: str = DEFAULT_LANGUAGE_VARIANT
    _source_header: str = ""   # prepended to render() output for provenance treatment
    _public_suffix: str = ""   # appended to render() output for public signal injection

    def render(self) -> str:
        """Render to the text the agent will see."""
        body = self._render_body()
        if self._source_header:
            body = self._source_header + "\n\n" + body
        if self._public_suffix:
            return body + self._public_suffix
        return body

    def _render_body(self) -> str:
        obs_text = "\n".join(f"  - {o}" for o in self.observations)
        unclear_text = "\n".join(f"  - {u}" for u in self.unclear)
        if self.language_variant == "legacy":
            return (
                f"BOTTOM LINE: {self.bottom_line}\n\n"
                f"OBSERVATIONS:\n{obs_text}\n\n"
                f"WHAT'S UNCLEAR:\n{unclear_text}\n\n"
                f"ATMOSPHERE: {self.atmosphere}"
            )
        if self.language_variant == "baseline_min":
            return (
                "BRIEFING\n\n"
                f"BOTTOM LINE: {self.bottom_line}\n"
                f"ATMOSPHERE: {self.atmosphere}\n\n"
                f"EVIDENCE:\n{obs_text}\n\n"
                f"UNCLEAR:\n{unclear_text}"
            )
        if self.language_variant == "baseline":
            return (
                "PRIVATE BRIEFING\n\n"
                f"BOTTOM LINE: {self.bottom_line}\n"
                f"ATMOSPHERE: {self.atmosphere}\n\n"
                f"EVIDENCE:\n{obs_text}\n\n"
                f"UNCERTAINTIES:\n{unclear_text}"
            )
        if self.language_variant == "baseline_assess":
            return (
                "PRIVATE INTELLIGENCE BRIEFING\n\n"
                "ASSESSMENT:\n"
                f"  - Regime outlook: {self.bottom_line}\n"
                f"  - Atmosphere: {self.atmosphere}\n"
                f"  - Signal clarity: {_clarity_label(self.clarity)}\n\n"
                f"EVIDENCE MOSAIC:\n{obs_text}\n\n"
                f"KEY UNCERTAINTIES:\n{unclear_text}"
            )
        if self.language_variant == "baseline_full":
            return (
                "PRIVATE INTELLIGENCE BRIEFING\n\n"
                "ASSESSMENT SNAPSHOT:\n"
                f"  - Regime outlook: {self.bottom_line}\n"
                f"  - Signal clarity: {_clarity_label(self.clarity)}\n"
                f"  - Coordination climate: {self.atmosphere}\n\n"
                f"EVIDENCE MOSAIC:\n{obs_text}\n\n"
                f"KEY UNCERTAINTIES:\n{unclear_text}\n\n"
                "Use this information only; you have no additional evidence."
            )
        if self.language_variant == "cable":
            # Terse diplomatic-cable style — no narrative framing, short lines
            obs_cable = "\n".join(f"  {i+1}. {o}" for i, o in enumerate(self.observations))
            unclear_cable = "\n".join(f"  - {u}" for u in self.unclear)
            return (
                f"SUBJ: SITUATION ASSESSMENT\n"
                f"CLASSIFICATION: PRIVATE\n\n"
                f"1. SUMMARY: {self.bottom_line}\n"
                f"2. ATMOSPHERE: {self.atmosphere}\n"
                f"3. REPORTING:\n{obs_cable}\n"
                f"4. GAPS:\n{unclear_cable}\n"
                f"5. END CABLE."
            )
        if self.language_variant == "journalistic":
            # News-wire inverted pyramid style — lead, then details
            obs_journal = "\n".join(f"  * {o}" for o in self.observations)
            unclear_journal = "\n".join(f"  * {u}" for u in self.unclear)
            return (
                f"FIELD REPORT\n\n"
                f"{self.bottom_line} "
                f"Sources describe the atmosphere as: {self.atmosphere.lower()}.\n\n"
                f"Key observations:\n{obs_journal}\n\n"
                f"Analysts note the following remain unclear:\n{unclear_journal}"
            )
        raise ValueError(f"Unsupported language_variant at render time: {self.language_variant}")


class BriefingGenerator:
    """Generates calibrated intelligence briefings from private signals.

    Parameters
    ----------
    cutoff_center : float
        Where the theoretical x* cutoff sits in z-score space.
        Shift this during calibration to move the empirical threshold.
    clarity_width : float
        Width of the ambiguous region. Larger = broader gray zone.
    direction_slope : float
        Steepness of direction logistic. Lower = more gradual transition
        between "weak regime" and "strong regime" briefings.
    coordination_slope : float
        Steepness of coordination logistic.
    dissent_floor : float
        Minimum fraction of contrary evidence in any briefing.
        0.25 means at least ~25% of cues contradict the dominant signal.
    mixed_cue_clarity : float
        Clarity threshold below which genuinely ambiguous cues appear.
    n_observations : int
        Number of evidence bullets per briefing.
    bottomline_cuts : sequence[float], optional
        Five cutpoints for mapping direction to bottom-line text tiers.
    unclear_cuts : sequence[float], optional
        Four cutpoints for mapping direction to uncertainty text tiers.
    coordination_cuts : sequence[float], optional
        Four cutpoints for mapping coordination to atmosphere text tiers.
    coordination_blend_prob : float
        Blend probability used in intermediate atmosphere bands.
    language_variant : str
        Rendering schema for the final briefing text.
        One of: "legacy", "baseline_min", "baseline", "baseline_assess",
        "baseline_full", "cable", "journalistic".
    seed : int
        Random seed for reproducibility within a period. Each agent
        should get a different seed offset.
    """

    def __init__(self, cutoff_center=0.0, clarity_width=1.0,
                 direction_slope=0.8, coordination_slope=0.6,
                 dissent_floor=0.25, mixed_cue_clarity=0.5,
                 n_observations=8,
                 bottomline_cuts=None,
                 unclear_cuts=None,
                 coordination_cuts=None,
                 coordination_blend_prob=DEFAULT_COORDINATION_BLEND_PROB,
                 language_variant=DEFAULT_LANGUAGE_VARIANT,
                 seed=None,
                 source_header="",
                 rhetoric_bias=0.0):
        self.cutoff_center = cutoff_center
        self.clarity_width = clarity_width
        self.direction_slope = direction_slope
        self.coordination_slope = coordination_slope
        self.dissent_floor = dissent_floor
        self.mixed_cue_clarity = mixed_cue_clarity
        self.n_observations = n_observations
        self.source_header = source_header
        self.rhetoric_bias = float(rhetoric_bias)
        self.bottomline_cuts = _validate_cutpoints(
            "bottomline_cuts",
            bottomline_cuts if bottomline_cuts is not None else DEFAULT_BOTTOMLINE_CUTS,
            expected_len=8,
        )
        self.unclear_cuts = _validate_cutpoints(
            "unclear_cuts",
            unclear_cuts if unclear_cuts is not None else DEFAULT_UNCLEAR_CUTS,
            expected_len=5,
        )
        self.coordination_cuts = _validate_cutpoints(
            "coordination_cuts",
            coordination_cuts if coordination_cuts is not None else DEFAULT_COORDINATION_CUTS,
            expected_len=5,
        )
        self.coordination_blend_prob = float(np.clip(coordination_blend_prob, 0.0, 1.0))
        self.language_variant = _validate_language_variant(language_variant)
        self.base_seed = seed

    def generate(self, z_score, agent_id=0, period=0):
        """Generate a briefing for one agent.

        Parameters
        ----------
        z_score : float
            Private signal in z-score units: (x_i - z) / sigma.
        agent_id : int
            Agent identifier (used for seed uniqueness).
        period : int
            Time period (used for seed uniqueness).

        Returns
        -------
        Briefing
        """
        # Unique RNG per agent-period so briefings are reproducible but distinct.
        seed = _seed_for_agent_period(self.base_seed, agent_id, period)
        rng = np.random.default_rng(seed)

        direction, clarity, coordination = _compute_sliders(
            z_score, self.cutoff_center, self.clarity_width,
            self.direction_slope, self.coordination_slope,
        )

        bottom_line = _pick_bottom_line(direction, rng, cuts=self.bottomline_cuts)

        # Sample observations from different domains (8 from 8)
        n_obs = min(self.n_observations, len(DOMAINS))
        domain_indices = rng.choice(len(DOMAINS), size=n_obs, replace=False)
        tagged_variants = {"baseline", "baseline_assess", "baseline_full", "cable"}
        if self.language_variant in tagged_variants:
            domain_indices = np.sort(domain_indices)
        observations = []
        for di in domain_indices:
            bullet = _sample_evidence_item(DOMAINS[di], direction, clarity, rng,
                                          self.dissent_floor, self.mixed_cue_clarity,
                                          rung_bias=self.rhetoric_bias)
            if self.language_variant in tagged_variants:
                domain_name = _format_domain_name(DOMAINS[di]["name"])
                bullet = f"[{domain_name}] {bullet}"
            observations.append(bullet)

        unclear = _pick_unclear_items(direction, rng, n_items=2, cuts=self.unclear_cuts)
        atmosphere = _pick_atmosphere(
            coordination,
            rng,
            cuts=self.coordination_cuts,
            blend_prob=self.coordination_blend_prob,
        )

        briefing = Briefing(
            bottom_line=bottom_line,
            observations=observations,
            unclear=unclear,
            atmosphere=atmosphere,
            z_score=z_score,
            direction=direction,
            clarity=clarity,
            coordination=coordination,
            language_variant=self.language_variant,
        )
        if self.source_header:
            briefing._source_header = self.source_header
        return briefing
