"""
Rule-based answer corruption for generating hallucinated answers.

Provides deterministic, seedable transformations that turn a correct answer
into a plausible-looking but factually wrong one.  **No LLM calls** — this
module is fully offline.

Three strategies are applied in sequence:
1. Numeric substitution — doubles or triples numbers.
2. Negation insertion — adds Hindi/Tamil negation markers.
3. Entity swap — swaps the first two named-entity-like tokens.

All randomness comes from ``random.Random(seed)`` so that repeated runs
with the same seed produce identical outputs.
"""

from __future__ import annotations

import logging
import random
import re
from typing import Optional

logger = logging.getLogger(__name__)

# Negation words per language.  Inserted before the main verb to flip meaning.
_NEGATION_MARKERS: dict[str, list[str]] = {
    "hindi": ["नहीं", "कभी नहीं"],
    "tamil": ["இல்லை", "மாட்டார்"],
    "marathi": ["नाही", "कधीच नाही"],
    "english": ["not", "never"],
}

# Multipliers used by numeric substitution, picked at random.
_NUM_MULTIPLIERS = [2, 3, 0.5, 10]


def _substitute_numbers(text: str, rng: random.Random) -> str:
    """Replace numeric values with plausible but wrong alternatives.

    Args:
        text: Input text potentially containing numbers.
        rng: Seeded ``random.Random`` instance for reproducibility.

    Returns:
        Text with numbers multiplied by a random factor.
    """

    def _replace(match: re.Match) -> str:
        original = match.group(0)
        try:
            value = float(original.replace(",", ""))
        except ValueError:
            return original
        multiplier = rng.choice(_NUM_MULTIPLIERS)
        new_value = value * multiplier
        # Preserve integer formatting when the original had no decimal.
        if "." not in original and new_value == int(new_value):
            return str(int(new_value))
        return f"{new_value:g}"

    # Match integers, decimals, and comma-separated numbers (e.g. 6,000).
    return re.sub(r"\d[\d,]*\.?\d*", _replace, text)


def _insert_negation(text: str, language: str, rng: random.Random) -> str:
    """Insert a negation marker to flip the factual claim.

    The negation word is inserted roughly in the middle of the sentence so
    that the result reads as a plausible (but wrong) statement.

    Args:
        text: Input text.
        language: Language key (must be in ``_NEGATION_MARKERS``).
        rng: Seeded ``random.Random`` instance.

    Returns:
        Text with a negation word inserted, or the original text unchanged
        if the language is not supported.
    """
    markers = _NEGATION_MARKERS.get(language)
    if not markers:
        logger.debug(
            "No negation markers for language '%s'; skipping negation", language
        )
        return text

    words = text.split()
    if not words:
        return text

    marker = rng.choice(markers)
    insert_pos = max(1, len(words) // 2)
    words.insert(insert_pos, marker)
    return " ".join(words)


def _swap_entities(text: str, rng: random.Random) -> str:
    """Swap the first two multi-character capitalized / Devanagari tokens.

    This is a rough heuristic that catches many named entities without
    requiring a full NER model.

    Args:
        text: Input text.
        rng: Seeded ``random.Random`` instance (unused currently, reserved
            for future randomization).

    Returns:
        Text with two entity-like tokens swapped, or the original if fewer
        than two candidates were found.
    """
    # Match tokens that look like named entities:
    # - Devanagari word of 3+ chars  OR
    # - Latin word starting with uppercase of 2+ chars
    pattern = re.compile(r"[\u0900-\u097F]{3,}|[A-Z][a-zA-Z]+")
    matches = list(pattern.finditer(text))
    if len(matches) < 2:
        return text

    # Pick the first two distinct matches.
    m1, m2 = matches[0], matches[1]
    if m1.group() == m2.group():
        return text

    # Swap via placeholder to avoid partial-replacement collisions.
    result = text[: m1.start()] + "<<SWAP_B>>" + text[m1.end() :]
    # Recalculate m2 position after first replacement.
    offset = len("<<SWAP_B>>") - (m1.end() - m1.start())
    result = (
        result[: m2.start() + offset]
        + m1.group()
        + result[m2.end() + offset :]
    )
    result = result.replace("<<SWAP_B>>", m2.group())
    return result


def corrupt_answer(
    answer: str,
    language: str = "hindi",
    seed: int = 42,
    strategies: Optional[list[str]] = None,
) -> str:
    """Generate a hallucinated version of a correct answer.

    Applies one or more rule-based corruption strategies to produce a
    plausible-but-wrong answer.  Fully deterministic for a given seed.

    Args:
        answer: The correct answer text.
        language: Language of the text (``"hindi"``, ``"tamil"``, etc.).
        seed: Random seed for reproducibility.
        strategies: Which strategies to apply.  Defaults to all three:
            ``["numeric", "negation", "entity_swap"]``.

    Returns:
        A corrupted version of the answer.

    Example:
        >>> corrupt_answer("योजना में 6000 रुपये मिलते हैं।", seed=42)
        'में योजना 12000 नहीं रुपये मिलते हैं।'
    """
    if strategies is None:
        strategies = ["numeric", "negation", "entity_swap"]

    rng = random.Random(seed)
    result = answer

    for strategy in strategies:
        if strategy == "numeric":
            result = _substitute_numbers(result, rng)
        elif strategy == "negation":
            result = _insert_negation(result, language, rng)
        elif strategy == "entity_swap":
            result = _swap_entities(result, rng)
        else:
            logger.warning("Unknown corruption strategy: '%s'", strategy)

    if result == answer:
        logger.warning(
            "corrupt_answer() could not change the answer %r (language=%r); "
            "no corruption strategy applied. This answer will be identical "
            "to the ground truth in benchmark results — treat any "
            "groundedness comparison for it as uninformative.",
            answer,
            language,
        )

    return result
