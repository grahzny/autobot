"""Epistemic Grounding -- heuristic detection of ungrounded assertions.

Runs on LLM output (thinking, response, notes) to detect claims that
aren't supported by the entity's known world state. Does NOT call the LLM.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class GroundingResult:
    """Result of running text through the grounding gate."""

    original: str
    flags: list[str] = field(default_factory=list)
    ungrounded_count: int = 0


# Patterns suggesting the LLM is asserting invented history
_MEMORY_CLAIM_PATTERNS = [
    re.compile(r"(?i)\bI remember\b.{5,}"),
    re.compile(r"(?i)\blast time\b.{5,}"),
    re.compile(r"(?i)\byou (?:told|said|mentioned)\b.{5,}"),
    re.compile(r"(?i)\bwe (?:talked|discussed|spoke)\b.{5,}"),
    re.compile(r"(?i)\byou once\b.{5,}"),
    re.compile(r"(?i)\bback when\b.{5,}"),
    re.compile(r"(?i)\bremember when\b.{5,}"),
]

# Patterns suggesting the LLM is asserting facts about the external world
_FACT_CLAIM_PATTERNS = [
    re.compile(r"(?i)\bI (?:know|learned|discovered|found out)\b.{5,}"),
    re.compile(r"(?i)\baccording to\b.{5,}"),
    re.compile(r"(?i)\bit turns out\b.{5,}"),
    re.compile(r"(?i)\bthe (?:fact|truth) is\b.{5,}"),
]

# Patterns suggesting ungrounded market predictions
_MARKET_CLAIM_PATTERNS = [
    re.compile(r"(?i)\b(?:the stock|it|the price) will\b.{5,}"),
    re.compile(r"(?i)\b(?:it's|it is) going to (?:rally|crash|drop|surge|moon)\b"),
    re.compile(r"(?i)\bearnings will (?:beat|miss|exceed)\b"),
    re.compile(r"(?i)\bguaranteed (?:return|profit|gain)\b"),
    re.compile(r"(?i)\bcan't (?:lose|go down|fail)\b"),
    re.compile(r"(?i)\b(?:definitely|certainly|absolutely) (?:buy|sell|going)\b"),
]


def check_grounding(
    text: str,
    known_people: set[str],
    known_topics: set[str],
    conversation_exists: bool = False,
) -> GroundingResult:
    """Check text for ungrounded assertions.

    Scans for memory claims ("I remember when...") and fact claims
    ("it turns out...") that aren't grounded in known people/topics.
    """
    flags: list[str] = []

    for pattern in _MEMORY_CLAIM_PATTERNS:
        match = pattern.search(text)
        if match:
            claim_text = match.group(0)
            # Check if claim references a known person or topic
            has_known_ref = any(
                name.lower() in claim_text.lower()
                for name in known_people | known_topics
            )
            if not has_known_ref and not conversation_exists:
                flags.append(f"memory_claim: {claim_text[:80]}")

    for pattern in _FACT_CLAIM_PATTERNS:
        match = pattern.search(text)
        if match:
            flags.append(f"fact_claim: {match.group(0)[:80]}")

    return GroundingResult(
        original=text,
        flags=flags,
        ungrounded_count=len(flags),
    )


# Words that indicate subjective observation (always allowed in notes)
_SUBJECTIVE_MARKERS = (
    "seems", "appears", "I think", "I feel", "maybe",
    "might", "could be", "I wonder", "probably", "possibly",
)

# Common stop words to ignore when checking note grounding
_STOP_WORDS = frozenset({
    "the", "a", "an", "is", "was", "are", "were", "they", "them",
    "their", "about", "with", "that", "this", "and", "or", "but",
    "for", "in", "on", "to", "of", "has", "had", "have", "be",
    "been", "being", "do", "does", "did", "not", "no", "so", "if",
})


def ground_notes(
    notes: list[str],
    known_people: set[str],
    conversation_messages: list[dict],
) -> list[str]:
    """Filter person notes, flagging ungrounded factual claims.

    A note is grounded if:
    1. It's a subjective observation ("seems", "appears", "I think")
    2. Its key content words appear in actual conversation messages

    Ungrounded factual claims are prefixed with "[unverified] ".
    """
    msg_text = " ".join(
        m.get("text", "") for m in conversation_messages
    ).lower()

    grounded: list[str] = []
    for note in notes:
        note_lower = note.lower()

        # Subjective notes always pass through
        if any(marker in note_lower for marker in _SUBJECTIVE_MARKERS):
            grounded.append(note)
            continue

        # Check if key content words from note appear in messages
        words = set(note_lower.split())
        content_words = words - _STOP_WORDS
        if not content_words:
            grounded.append(note)
            continue

        overlap = sum(1 for w in content_words if w in msg_text)
        if overlap >= 2 or len(content_words) <= 2:
            grounded.append(note)
        else:
            grounded.append(f"[unverified] {note}")

    return grounded


def check_market_grounding(
    text: str,
    known_prices: dict[str, float] | None = None,
    known_fundamentals: dict[str, dict] | None = None,
) -> GroundingResult:
    """Check text for ungrounded market assertions.

    Flags speculative predictions that claim certainty, and price claims
    that don't match known data.
    """
    flags: list[str] = []

    for pattern in _MARKET_CLAIM_PATTERNS:
        match = pattern.search(text)
        if match:
            flags.append(f"market_prediction: {match.group(0)[:80]}")

    # Check price claims against known data
    if known_prices:
        price_pattern = re.compile(
            r"(?i)(?:trading at|priced at|currently at|worth)\s*\$?([\d,]+(?:\.\d+)?)"
        )
        for match in price_pattern.finditer(text):
            claimed_price = float(match.group(1).replace(",", ""))
            # Check if any known price is close
            near_known = any(
                abs(claimed_price - p) / p < 0.05  # within 5%
                for p in known_prices.values()
                if p > 0
            )
            if not near_known and claimed_price > 0:
                flags.append(f"price_claim: ${claimed_price:.2f} (not near any known price)")

    return GroundingResult(
        original=text,
        flags=flags,
        ungrounded_count=len(flags),
    )
