"""
Voyager-style BT library for spark_real.

Persists behaviour-tree scores from real-robot trials to disk and serves them
back, either as few-shot examples for the planner or as the executed plan
itself when the LLM is disabled.

Layout::

    bt_library/
        index.json           -> [{hash, instruction, success, fail}, ...]
        runtime_state.json   -> {use_cached_bt, pins}
        <hash>.json          -> {instruction, score, objects, success, fail, ...}

The hash is content-addressed (instruction + canonicalised score) so the
same plan executed twice merges into one entry with bumped success/fail
counters.

Retrieval is *not* fuzzy-first. ``lookup()`` resolves in this order:

    1. per-task pin  -- an operator/registry binding of one instruction to
       one entry hash; persisted, and NOT consumed on read, so N consecutive
       runs of the same task serve the same tree.
    2. exact match   -- normalised instruction equality against an entry's
       ``instruction`` or any of its ``aliases``.
    3. similarity    -- Jaccard over content words, gated by
       ``min_similarity`` (default 0.60). Below the floor is a MISS, not a
       hit (at ``sim > 0``, "put the pen in the bin" matched a tray plan at
       0.20).

Usage::

    lib = BTLibrary(Path("output/bt_library"))
    match = lib.lookup("put the knife in the tray")
    if match is not None:
        score, source = match.score, match.source   # source: pin|exact|alias|similar
    examples = lib.top_k_for(instruction, k=3)      # -> List[(instruction, score)]
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from spark_real.bt_keys import (  # noqa: F401
    _DEFAULT_PARAMS,
    _NEAR_EXACT_FILLER,
    _NONSEMANTIC_KEYS,
    _POSITIONAL_PREPOSITIONS,
    _SPELLING_VARIANTS,
    _STOPWORDS,
    DEFAULT_AUTO_PROMOTE_AFTER,
    DEFAULT_MIN_SIMILARITY,
    BTCacheMiss,
    BTEntry,
    BTMatch,
    _canon_word,
    _canonical_score_str,
    _hash_entry,
    _is_order_sensitive,
    _jaccard,
    _near_exact_key,
    _norm,
    _ordered_tokens,
    _singular,
    _strip_nonsemantic,
    _tokens,
    near_exact_key,
    normalize_instruction,
)
from spark_real.bt_retrieval import BTRetrievalMixin  # noqa: F401

logger = logging.getLogger(__name__)


class BTLibrary(BTRetrievalMixin):
    """
    Disk-backed library of BT scores keyed by instruction.
    """

    def __init__(
        self,
        root: Path,
        *,
        min_similarity: float = DEFAULT_MIN_SIMILARITY,
        auto_promote_after: int = DEFAULT_AUTO_PROMOTE_AFTER,
    ):
        self.root = Path(root).expanduser()
        self.root.mkdir(parents=True, exist_ok=True)
        self.index_path = self.root / "index.json"
        self.state_path = self.root / "runtime_state.json"
        self.min_similarity = float(min_similarity)
        self.auto_promote_after = int(auto_promote_after)
        self._entries: Dict[str, BTEntry] = {}
        # Persisted runtime state (survives server restarts).
        self._pins: Dict[str, str] = {}
        self._use_cached_bt: Optional[bool] = None
        self._load()
        self._load_runtime_state()

    # I/O

    def _load(self) -> None:
        if not self.index_path.exists():
            return
        try:
            blob = json.loads(self.index_path.read_text())
        except (OSError, json.JSONDecodeError) as e:
            logger.warning("BTLibrary: bad index %s: %s; starting fresh", self.index_path, e)
            return
        for stub in blob:
            h = stub.get("hash")
            if not h:
                continue
            entry_path = self.root / f"{h}.json"
            if not entry_path.exists():
                continue
            try:
                self._entries[h] = BTEntry.from_dict(json.loads(entry_path.read_text()))
            except Exception as e:
                logger.warning("BTLibrary: skipping %s: %s", entry_path, e)

    def _write_entry(self, entry: BTEntry) -> None:
        (self.root / f"{entry.hash}.json").write_text(json.dumps(entry.to_dict(), indent=2))

    def _flush(self) -> None:
        index = [
            {
                "hash": e.hash,
                "instruction": e.instruction,
                "success": e.success,
                "fail": e.fail,
                "promoted": e.promoted,
                "seed": e.seed,
            }
            for e in self._entries.values()
        ]
        self.index_path.write_text(json.dumps(index, indent=2))

    # persisted runtime state

    def _load_runtime_state(self) -> None:
        if not self.state_path.exists():
            return
        try:
            blob = json.loads(self.state_path.read_text())
        except (OSError, json.JSONDecodeError) as e:
            logger.warning("BTLibrary: bad runtime state %s: %s", self.state_path, e)
            return
        raw_pins = blob.get("pins") or {}
        self._pins = {_norm(k): str(v) for k, v in raw_pins.items() if _norm(k) and v}
        cached = blob.get("use_cached_bt")
        self._use_cached_bt = None if cached is None else bool(cached)

    def _flush_runtime_state(self) -> None:
        self.state_path.write_text(
            json.dumps(
                {"use_cached_bt": self._use_cached_bt, "pins": self._pins},
                indent=2,
            )
        )

    @property
    def use_cached_bt(self) -> Optional[bool]:
        """
        Persisted cache-only override, or None to defer to env/config.
        """
        return self._use_cached_bt

    def set_use_cached_bt(self, value: Optional[bool]) -> Optional[bool]:
        self._use_cached_bt = None if value is None else bool(value)
        self._flush_runtime_state()
        return self._use_cached_bt

    # per-task pins

    def pins(self) -> Dict[str, str]:
        """
        Normalised instruction -> entry hash. Copy; mutate via pin/unpin.
        """
        return dict(self._pins)

    def pin(self, instruction: str, h: str) -> Optional[BTEntry]:
        """
        Bind one instruction to one entry. Persisted, and not consumed on
        read, so 30 consecutive runs of a task serve the same tree.
        """
        entry = self._entries.get(h)
        if entry is None:
            return None
        key = _norm(instruction)
        if not key:
            return None
        self._pins[key] = h
        self._flush_runtime_state()
        return entry

    def unpin(self, instruction: str) -> Optional[str]:
        prev = self._pins.pop(_norm(instruction), None)
        if prev is not None:
            self._flush_runtime_state()
        return prev

    def pinned_for(self, instruction: str) -> Optional[BTEntry]:
        h = self._pins.get(_norm(instruction))
        if h is None:
            return None
        entry = self._entries.get(h)
        if entry is None:
            # Pin points at a deleted entry: drop it rather than silently
            # falling through on every future call.
            self._pins.pop(_norm(instruction), None)
            self._flush_runtime_state()
            logger.warning(
                "BTLibrary: pin for %r referenced missing entry %s; cleared",
                instruction,
                h,
            )
        return entry

    # promotion

    def set_promoted(self, h: str, promoted: bool) -> Optional[BTEntry]:
        """
        Toggle the ``promoted`` flag on an entry by hash.

        Promoted entries are always injected into the planner prompt
        (independent of jaccard similarity to the current instruction),
        so they act as canonical recipes the LLM can rely on. Regular
        entries only show up when the current instruction is similar
        to the entry's instruction.
        """
        entry = self._entries.get(h)
        if entry is None:
            return None
        entry.promoted = bool(promoted)
        self._write_entry(entry)
        self._flush()
        return entry

    def promoted_entries(self) -> List[Tuple[str, Any]]:
        """
        Return (instruction, score) for every promoted entry. Used
        by the planner to always-include these as few-shot examples
        in addition to similarity-retrieved ones.
        """
        return [
            (e.instruction, e.score)
            for e in self._entries.values()
            if e.promoted and e.success >= 1
        ]

    def list_all(self) -> List[Dict[str, Any]]:
        """
        Index view for the frontend library panel.
        """
        pinned = set(self._pins.values())
        out = []
        for e in self._entries.values():
            out.append(
                {
                    "hash": e.hash,
                    "instruction": e.instruction,
                    "aliases": e.aliases,
                    "objects": e.objects,
                    "success": e.success,
                    "fail": e.fail,
                    "promoted": e.promoted,
                    "seed": e.seed,
                    "priority": e.priority,
                    "verified": e.verified,
                    "notes": e.notes,
                    "pinned": e.hash in pinned,
                }
            )
        out.sort(key=lambda x: (not x["seed"], not x["promoted"], -x["success"]))
        return out

    def get(self, h: str) -> Optional[BTEntry]:
        return self._entries.get(h)

    def get_score(self, h: str) -> Optional[Any]:
        e = self._entries.get(h)
        return e.score if e is not None else None

    def entries(self) -> Iterable[BTEntry]:
        return self._entries.values()

    # mutation

    def add(
        self,
        instruction: str,
        score: Any,
        objects: Optional[List[str]] = None,
        success: bool = True,
        aliases: Optional[List[str]] = None,
        seed: bool = False,
        priority: int = 0,
        notes: str = "",
        verified: bool = True,
    ) -> BTEntry:
        """
        Insert or update an entry; bumps the success/fail counter.

        Callers that *served* a plan from this library must use ``bump()``
        instead: re-adding a cache hit under a slightly different
        instruction string mints a near-duplicate entry.
        """
        h = _hash_entry(instruction, score)
        if h in self._entries:
            entry = self._entries[h]
            if success:
                entry.success += 1
            else:
                entry.fail += 1
            # Seed metadata is authoritative when re-seeding an existing row.
            if seed:
                entry.seed = True
                entry.priority = priority
                entry.notes = notes or entry.notes
                entry.verified = verified
                for alias in aliases or []:
                    if alias not in entry.aliases:
                        entry.aliases.append(alias)
        else:
            entry = BTEntry(
                hash=h,
                instruction=instruction,
                score=score,
                objects=list(objects or []),
                success=1 if success else 0,
                fail=0 if success else 1,
                aliases=list(aliases or []),
                seed=bool(seed),
                priority=int(priority),
                notes=str(notes or ""),
                verified=bool(verified),
            )
            self._entries[h] = entry
        if not seed:
            self._maybe_auto_promote(entry)
        self._write_entry(entry)
        self._flush()
        return entry

    def bump(
        self,
        h: str,
        success: bool,
        alias: Optional[str] = None,
    ) -> Optional[BTEntry]:
        """
        Record an outcome against an entry that already exists, without
        minting a new row. This is the write path for every run whose plan
        was SERVED from the library (pin / exact / normalized / similar).

        ``add()`` would instead mint a near-duplicate keyed on the run's
        phrasing (the live library once grew to 88 rows for ~20 distinct
        tasks). Passing the run's phrasing as ``alias`` records it on the
        entry that ran, so the next run of that phrasing is an exact hit.
        """
        entry = self._entries.get(h)
        if entry is None:
            return None
        if success:
            entry.success += 1
        else:
            entry.fail += 1
        if alias:
            self._note_alias(entry, alias)
        self._maybe_auto_promote(entry)
        self._write_entry(entry)
        self._flush()
        return entry

    def note_unverified(
        self,
        h: Optional[str],
        instruction: Optional[str] = None,
        score: Any = None,
        objects: Optional[List[str]] = None,
    ) -> Optional[BTEntry]:
        """
        Record that a run of this entry produced NO verdict.

        Bumps neither success nor fail: an unverified run must not reinforce a
        tree and must not penalise one either. It makes the verifier's blind
        spots countable on /scores.

        Mints the row when the hash is unknown and the run's material is
        supplied, so an LLM-planned run (no ``bt_hash`` yet) that the verifier
        never resolves still accumulates a counter. A minted row carries
        success=0, below ``lookup``'s ``min_success=1`` floor, so it is
        countable without becoming cache-eligible.
        """
        entry = self._entries.get(h) if h else None
        if entry is None:
            if not instruction or not score:
                return None
            h2 = _hash_entry(instruction, score)
            entry = self._entries.get(h2)
            if entry is None:
                entry = BTEntry(
                    hash=h2,
                    instruction=instruction,
                    score=score,
                    objects=list(objects or []),
                )
                self._entries[h2] = entry
        entry.unverified += 1
        self._write_entry(entry)
        self._flush()
        return entry

    def _note_alias(self, entry: BTEntry, phrasing: str) -> bool:
        """
        Record an extra instruction spelling on an entry, if it is not
        already one of its keys. Returns True when something was added.
        """
        key = _norm(phrasing)
        if not key or key in entry.keys():
            return False
        entry.aliases.append(str(phrasing))
        logger.info(
            "BTLibrary: %s learned alias %r (now %d spelling(s))",
            entry.hash,
            phrasing,
            len(entry.keys()),
        )
        return True

    def alias(self, h: str, phrasing: str) -> Optional[BTEntry]:
        """
        Bind an extra instruction spelling to an existing entry.
        """
        entry = self._entries.get(h)
        if entry is None:
            return None
        if self._note_alias(entry, phrasing):
            self._write_entry(entry)
            self._flush()
        return entry

    def _maybe_auto_promote(self, entry: BTEntry) -> bool:
        """
        Promote an entry that has proven itself. Caller flushes.
        """
        if entry.promoted or self.auto_promote_after <= 0:
            return False
        if entry.success >= self.auto_promote_after and entry.fail == 0:
            entry.promoted = True
            logger.info(
                "BTLibrary: auto-promoted %s (%r) at %d/%d successes",
                entry.hash,
                entry.instruction,
                entry.success,
                self.auto_promote_after,
            )
            return True
        return False

    def delete_by_instruction(self, instruction: str) -> List[str]:
        """
        Delete every entry keyed on ``instruction`` (exact or alias).
        Returns the hashes removed. Pruning tool for the operator: a task
        whose cached tree turned out to be wrong is one call to clear.
        """
        doomed = [e.hash for e in self.exact_matches(instruction, min_success=0)]
        for h in doomed:
            self.delete(h)
        return doomed

    def delete(self, h: str) -> Optional[BTEntry]:
        """
        Remove an entry from disk and the in-memory index, and drop any
        pins that referenced it.
        """
        entry = self._entries.pop(h, None)
        if entry is None:
            return None
        path = self.root / f"{h}.json"
        if path.exists():
            path.unlink()
        self._flush()
        stale = [k for k, v in self._pins.items() if v == h]
        if stale:
            for k in stale:
                self._pins.pop(k, None)
            self._flush_runtime_state()
        return entry

    def __len__(self) -> int:
        return len(self._entries)

    def stats(self) -> Dict[str, int]:
        total_success = sum(e.success for e in self._entries.values())
        total_fail = sum(e.fail for e in self._entries.values())
        return {
            "entries": len(self._entries),
            "seeds": sum(1 for e in self._entries.values() if e.seed),
            "pins": len(self._pins),
            "total_success": total_success,
            "total_fail": total_fail,
        }


__all__ = [
    "BTLibrary",
    "BTEntry",
    "BTMatch",
    "BTCacheMiss",
    "DEFAULT_MIN_SIMILARITY",
    "DEFAULT_AUTO_PROMOTE_AFTER",
    "normalize_instruction",
    "near_exact_key",
]
