"""BT library lookup: pin, exact, near-exact and similarity tiers, and few-shot top-k."""

from __future__ import annotations

import logging
from typing import Any, List, Optional, Tuple

from spark_real.bt_keys import (
    BTCacheMiss,
    BTEntry,
    BTMatch,
    _is_order_sensitive,
    _jaccard,
    _near_exact_key,
    _norm,
    _ordered_tokens,
    count_differs,
)

logger = logging.getLogger(__name__)


class BTRetrievalMixin:
    """BT library lookup: pin, exact, near-exact and similarity tiers, and few-shot top-k."""

    def _rank_key(self, entry: BTEntry) -> Tuple:
        """
        Total, stable order among equally-matching entries.

        Trailing hash makes it a total order; without it, ties fall back to
        index.json insertion order and the tree served for a task can flip
        mid-session.
        """
        return (
            entry.priority,
            int(entry.promoted),
            entry.success,
            -entry.fail,
            entry.hash,
        )

    def exact_matches(self, instruction: str, min_success: int = 0) -> List[BTEntry]:
        """
        Every entry whose instruction or alias normalises to ``instruction``,
        best first.
        """
        key = _norm(instruction)
        if not key:
            return []
        hits = [e for e in self._entries.values() if key in e.keys() and e.success >= min_success]
        hits.sort(key=self._rank_key, reverse=True)
        return hits

    def near_exact_matches(self, instruction: str, min_success: int = 0) -> List[BTEntry]:
        """
        Entries matching on the near-exact key: spelling variants, plurals
        and phrasing filler folded away, token ORDER preserved.

        This is the tier that absorbs everyday instruction variance --
        "put the spoons in the tray" / "pick up the spoon and place it in
        the tray" / "put the spoon in the tray" are one task -- without
        letting Jaccard merge genuinely different ones.
        """
        key = _near_exact_key(instruction)
        if not key:
            return []
        hits = [
            e
            for e in self._entries.values()
            if key in e.near_keys()
            and e.success >= min_success
            and not count_differs(instruction, e.keys())
        ]
        hits.sort(key=self._rank_key, reverse=True)
        return hits

    def lookup(
        self,
        instruction: str,
        min_success: int = 1,
        min_similarity: Optional[float] = None,
    ) -> Optional[BTMatch]:
        """
        Resolve an instruction to a single BT. Four tiers, best first:

            1. ``pin``        -- persisted per-task binding, never consumed.
            2. ``exact``/``alias`` -- normalised string equality.
            3. ``normalized`` -- near-exact key (spelling, plurals, filler).
            4. ``similar``    -- Jaccard >= floor, gated on token ORDER when
               the instruction contains a positional preposition.

        None on a miss. Below the similarity floor is a MISS, not a hit.
        """
        entry = self.pinned_for(instruction)
        if entry is not None:
            return BTMatch(entry=entry, source="pin", similarity=1.0)

        key = _norm(instruction)
        for hit in self.exact_matches(instruction, min_success=min_success):
            source = "exact" if _norm(hit.instruction) == key else "alias"
            return BTMatch(entry=hit, source=source, similarity=1.0)

        for hit in self.near_exact_matches(instruction, min_success=min_success):
            return BTMatch(entry=hit, source="normalized", similarity=1.0)

        floor = self.min_similarity if min_similarity is None else float(min_similarity)
        order_sensitive = _is_order_sensitive(instruction)
        query_ordered = _ordered_tokens(instruction)
        query = set(query_ordered)
        best: Optional[Tuple[float, Tuple, BTEntry]] = None
        for candidate in self._entries.values():
            if candidate.success < min_success:
                continue
            sim = _jaccard(query, set(_ordered_tokens(candidate.instruction)))
            if sim < floor:
                continue
            # Order gate: a set-similarity hit on an order-sensitive
            # instruction (see the positional-preposition list above) is only
            # trustworthy when the ordered token lists agree as well.
            if order_sensitive or _is_order_sensitive(candidate.instruction):
                if query_ordered != _ordered_tokens(candidate.instruction):
                    logger.debug(
                        "BTLibrary: rejecting order-sensitive fuzzy match "
                        "%r ~ %r (sim=%.2f, token order differs)",
                        instruction,
                        candidate.instruction,
                        sim,
                    )
                    continue
            ranked = (sim, self._rank_key(candidate), candidate)
            if best is None or (ranked[0], ranked[1]) > (best[0], best[1]):
                best = ranked
        if best is None:
            return None
        return BTMatch(entry=best[2], source="similar", similarity=best[0])

    def resolve(
        self,
        instruction: str,
        require_cached: bool = False,
        min_success: int = 1,
        min_similarity: Optional[float] = None,
    ) -> Optional[BTMatch]:
        """
        ``lookup`` plus strict offline mode.

        With ``require_cached`` a miss raises ``BTCacheMiss`` instead of
        letting the caller fall through to the LLM. Demo collection runs
        with this on: a silent LLM call mid-collection produces an
        episode whose plan nobody froze.
        """
        match = self.lookup(
            instruction,
            min_success=min_success,
            min_similarity=min_similarity,
        )
        if match is None and require_cached:
            raise BTCacheMiss(
                f"no cached BT for {instruction!r} "
                f"(floor={self.min_similarity:.2f}, entries={len(self._entries)}); "
                "seed one into the library or clear bt.require_cached"
            )
        return match

    def top_k_for(
        self,
        instruction: str,
        k: int = 3,
        min_success: int = 1,
        min_similarity: Optional[float] = None,
    ) -> List[Tuple[str, Any]]:
        """
        Up to k (instruction, score) pairs for few-shot prompting, best
        first. Exact/alias matches always outrank similarity matches.

        ``min_similarity=0.0`` disables the floor (any shared content word
        matches); the default applies it.
        """
        floor = self.min_similarity if min_similarity is None else float(min_similarity)
        picked: List[BTEntry] = []
        seen: set = set()
        for hit in self.exact_matches(instruction, min_success=min_success):
            picked.append(hit)
            seen.add(hit.hash)
        for hit in self.near_exact_matches(instruction, min_success=min_success):
            if hit.hash not in seen:
                picked.append(hit)
                seen.add(hit.hash)

        query = set(_ordered_tokens(instruction))
        scored: List[Tuple[float, Tuple, BTEntry]] = []
        for entry in self._entries.values():
            if entry.hash in seen or entry.success < min_success:
                continue
            sim = _jaccard(query, set(_ordered_tokens(entry.instruction)))
            if sim < floor or sim <= 0:
                continue
            scored.append((sim, self._rank_key(entry), entry))
        scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
        picked.extend(e for _, _, e in scored)
        return [(e.instruction, e.score) for e in picked[:k]]
