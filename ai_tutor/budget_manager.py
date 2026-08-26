"""
budget_manager.py
-----------------
Token-budget-aware prompt assembly for the AI Tutor pipeline.

BudgetManager.assemble() accepts named sections and per-section token budgets,
applies section-specific compression strategies when a section exceeds its
allocation, then joins the sections into a final prompt string.

Section-specific compression strategies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- ``learner_state``         Keep only the N lowest-mastery concepts plus all
                            misconceptions (N = budget_min of the section).
- ``conversation_history``  Fall back to a ``conversation_summary`` key that
                            must also be present in *sections* when the full
                            history is over budget.
- ``rag_knowledge``         Drop lowest-relevance chunks first (chunks are
                            assumed to arrive pre-sorted highest -> lowest
                            relevance, so we drop from the tail).

Any other section is compressed by hard-truncating its text at the token limit.

Token counting
~~~~~~~~~~~~~~
Uses a lightweight word-based approximation (1 token ~= 0.75 words) so the
module has zero external dependencies.  Swap ``_count_tokens`` for a tiktoken
call if you need exact counts.
"""

import warnings
import logging
import json
from typing import Dict, Tuple

logger = logging.getLogger("ai_tutor.budget_manager")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _count_tokens(text: str) -> int:
    """Approximate token count: 1 token ~= 0.75 words (GPT-style heuristic)."""
    if not text:
        return 0
    word_count = len(text.split())
    return max(1, round(word_count / 0.75))


def _truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Hard-truncate *text* so that its token count <= *max_tokens*."""
    words = text.split()
    target_words = max(1, round(max_tokens * 0.75))
    if len(words) <= target_words:
        return text
    return " ".join(words[:target_words]) + " ...[truncated]"


# ---------------------------------------------------------------------------
# Section-specific compression strategies
# ---------------------------------------------------------------------------

def _compress_learner_state(text: str, budget: Tuple[int, int]) -> str:
    """
    Keep only the N lowest-mastery concepts + all misconceptions.
    N is derived from budget[0] (the soft floor / budget_min).

    Expected *text* format — a JSON string with keys:
        - ``concepts``: list of {"name": str, "mastery": float}
        - ``misconceptions``: list of str   (always kept verbatim)

    Falls back to hard-truncation if the text is not valid JSON.
    """
    budget_min, budget_max = budget
    n_keep = max(1, budget_min // 20)  # rough: ~20 tokens per concept entry

    try:
        data = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        logger.debug("learner_state is not JSON; falling back to truncation.")
        return _truncate_to_tokens(text, budget_max)

    concepts: list = data.get("concepts", [])
    misconceptions: list = data.get("misconceptions", [])

    # Sort ascending by mastery score -> lowest mastery first
    sorted_concepts = sorted(
        concepts,
        key=lambda c: float(c.get("mastery", 1.0))
    )
    kept_concepts = sorted_concepts[:n_keep]

    compressed = {
        "concepts": kept_concepts,
        "misconceptions": misconceptions,
        "_note": f"Compressed: kept {len(kept_concepts)}/{len(concepts)} lowest-mastery concepts."
    }
    compressed_text = json.dumps(compressed, indent=2)

    # If still over budget, truncate the raw string
    if _count_tokens(compressed_text) > budget_max:
        compressed_text = _truncate_to_tokens(compressed_text, budget_max)

    return compressed_text


def _compress_conversation_history(
    text: str,
    budget: Tuple[int, int],
    sections: Dict[str, str]
) -> str:
    """
    Fall back to the ``conversation_summary`` section when the full history
    exceeds the budget.  The summary key must exist in *sections*; if it is
    absent or itself over budget the history is hard-truncated instead.
    """
    _, budget_max = budget

    summary = sections.get("conversation_summary", "")
    if summary:
        summary_tokens = _count_tokens(summary)
        if summary_tokens <= budget_max:
            logger.debug(
                "conversation_history over budget - using conversation_summary "
                "(%d tokens).", summary_tokens
            )
            return summary
        else:
            logger.debug(
                "conversation_summary also over budget (%d > %d); truncating.",
                summary_tokens, budget_max
            )
            return _truncate_to_tokens(summary, budget_max)
    else:
        logger.debug(
            "No conversation_summary available; truncating conversation_history."
        )
        return _truncate_to_tokens(text, budget_max)


def _compress_rag_knowledge(text: str, budget: Tuple[int, int]) -> str:
    """
    Drop the lowest-relevance chunks first.

    Expected format: chunks separated by ``\\n---\\n`` (markdown HR), already
    sorted highest -> lowest relevance.  We greedily keep chunks from the top
    until the token budget is exhausted.

    Falls back to hard-truncation for plain-text rag sections.
    """
    _, budget_max = budget
    separator = "\n---\n"

    if separator in text:
        chunks = text.split(separator)
        kept: list = []
        running_tokens = 0
        for chunk in chunks:
            chunk_tokens = _count_tokens(chunk)
            if running_tokens + chunk_tokens <= budget_max:
                kept.append(chunk)
                running_tokens += chunk_tokens
            else:
                # Budget exhausted - drop this and all subsequent (lower relevance) chunks
                dropped = len(chunks) - len(kept)
                logger.debug(
                    "rag_knowledge: dropped %d lowest-relevance chunk(s) to fit budget.",
                    dropped
                )
                break

        result = separator.join(kept)
        if not kept:
            # Even the highest-relevance chunk is over budget - truncate it
            result = _truncate_to_tokens(chunks[0], budget_max)
        return result
    else:
        # Plain text fallback
        return _truncate_to_tokens(text, budget_max)


# ---------------------------------------------------------------------------
# BudgetManager
# ---------------------------------------------------------------------------

class BudgetManager:
    """
    Token-budget-aware prompt assembler for the AI Tutor pipeline.

    Parameters
    ----------
    overall_ceiling : int, optional
        Hard ceiling on total assembled prompt tokens.  A ``UserWarning`` is
        raised if the assembled prompt still exceeds this after per-section
        compression.  Defaults to 8 000 tokens.

    section_separator : str, optional
        String used to join sections in the final prompt.
        Defaults to ``"\\n\\n"`` (double newline).

    Example
    -------
    >>> import json
    >>> bm = BudgetManager(overall_ceiling=4096)
    >>> sections = {
    ...     "system":               "You are a Socratic AI tutor...",
    ...     "learner_state":        json.dumps({
    ...                                 "concepts": [{"name": "Gradient Descent", "mastery": 0.2}],
    ...                                 "misconceptions": ["Confuses loss with accuracy"]
    ...                             }),
    ...     "rag_knowledge":        "Chunk A (high relevance)\\n---\\nChunk B (low relevance)",
    ...     "conversation_history": "Student: explain backprop\\nAssistant: Let's trace it...",
    ...     "conversation_summary": "Student asked about backprop.",
    ... }
    >>> budgets = {
    ...     "system":               (200, 400),
    ...     "learner_state":        (50,  200),
    ...     "rag_knowledge":        (300, 800),
    ...     "conversation_history": (100, 600),
    ... }
    >>> prompt_text, total_tokens = bm.assemble(sections, budgets)
    """

    def __init__(
        self,
        overall_ceiling: int = 8_000,
        section_separator: str = "\n\n"
    ):
        self.overall_ceiling = overall_ceiling
        self.section_separator = section_separator

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def assemble(
        self,
        sections: Dict[str, str],
        budgets: Dict[str, Tuple[int, int]]
    ) -> Tuple[str, int]:
        """
        Assemble and budget-compress a prompt from named sections.

        Parameters
        ----------
        sections : dict[str, str]
            Mapping of section name -> raw text content.  Sections are
            included in insertion order (Python 3.7+).
            The special key ``"conversation_summary"`` is treated as a
            fallback for ``"conversation_history"`` and is **not** emitted
            as its own section in the final prompt.

        budgets : dict[str, tuple[int, int]]
            Mapping of section name -> ``(budget_min, budget_max)``.
            *budget_min* is a soft lower bound used as a strategy parameter
            (e.g. number of concepts to keep); *budget_max* is the hard upper
            bound in tokens.  Sections absent from *budgets* are included
            verbatim without compression.

        Returns
        -------
        tuple[str, int]
            ``(assembled_prompt, total_token_count)``

        Warns
        -----
        UserWarning
            If the total token count still exceeds ``self.overall_ceiling``
            after all per-section compression has been applied.
        """
        # "conversation_summary" is a helper key only, never emitted directly
        _INTERNAL_KEYS = {"conversation_summary"}

        assembled_parts: list = []
        total_tokens = 0

        for section_name, raw_text in sections.items():
            if section_name in _INTERNAL_KEYS:
                continue

            budget = budgets.get(section_name)

            if budget is None:
                # No budget defined -> include verbatim
                final_text = raw_text
            else:
                budget_min, budget_max = budget
                section_tokens = _count_tokens(raw_text)

                if section_tokens <= budget_max:
                    final_text = raw_text
                else:
                    logger.info(
                        "Section '%s' exceeds budget (%d > %d tokens). "
                        "Applying compression strategy.",
                        section_name, section_tokens, budget_max
                    )
                    final_text = self._compress_section(
                        section_name=section_name,
                        text=raw_text,
                        budget=budget,
                        sections=sections
                    )

            if final_text:
                assembled_parts.append(final_text)
                total_tokens += _count_tokens(final_text)

        assembled_prompt = self.section_separator.join(assembled_parts)

        # Global ceiling check
        if total_tokens > self.overall_ceiling:
            warnings.warn(
                f"Assembled prompt total ({total_tokens} tokens) still exceeds "
                f"the overall ceiling of {self.overall_ceiling} tokens after "
                f"per-section compression. Consider tightening individual "
                f"budgets or raising the ceiling.",
                UserWarning,
                stacklevel=2
            )
            logger.warning(
                "Prompt ceiling breach: %d tokens > ceiling %d.",
                total_tokens, self.overall_ceiling
            )

        return assembled_prompt, total_tokens

    # ------------------------------------------------------------------
    # Internal dispatch
    # ------------------------------------------------------------------

    def _compress_section(
        self,
        section_name: str,
        text: str,
        budget: Tuple[int, int],
        sections: Dict[str, str]
    ) -> str:
        """Dispatch to the appropriate section-specific compression strategy."""
        if section_name == "learner_state":
            return _compress_learner_state(text, budget)
        elif section_name == "conversation_history":
            return _compress_conversation_history(text, budget, sections)
        elif section_name == "rag_knowledge":
            return _compress_rag_knowledge(text, budget)
        else:
            # Generic fallback: hard-truncate at the token limit
            _, budget_max = budget
            logger.debug(
                "No specific strategy for section '%s'; applying hard truncation.",
                section_name
            )
            return _truncate_to_tokens(text, budget_max)
