"""BT library keys and records: instruction normalization, near-exact keys, content hashes, entry types."""

from __future__ import annotations

import copy
import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List

# Similarity floor for the fuzzy fallback. Overridable per-library via the
# constructor (INTEGRATOR feeds ``bt.min_similarity`` from the family yaml).
#
# Deliberately high. Instruction variance is meant to be absorbed by the
# exact and near-exact tiers (aliases + spelling/plural canonicalisation),
# NOT by Jaccard: at a low floor "put the pen in the bin" scores 0.20
# against a tray plan and would execute it.
DEFAULT_MIN_SIMILARITY = 0.75

# Successes (with zero failures) after which an entry auto-promotes.
# Overridable per-library; INTEGRATOR feeds ``bt.auto_promote_after``.
DEFAULT_AUTO_PROMOTE_AFTER = 3

# Spelling / regional variants folded together before near-exact matching.
# Applied token-wise, so "grey block" and "gray block" are one cache key.
# Keep this list short and unambiguous -- it is not a synonym engine.
_SPELLING_VARIANTS = {
    "grey": "gray",
    "colour": "color",
    "coloured": "colored",
    "cutlery": "silverware",
    "utensil": "silverware",
    "cube": "block",
}

# Filler that never changes which objects a task touches. Dropped by the
# near-exact key so "pick up the knife and place it in the tray" and
# "put the knife in the tray" collapse to the same key.
_NEAR_EXACT_FILLER = frozenset(
    {
        "the",
        "a",
        "an",
        "to",
        "in",
        "into",
        "onto",
        "on",
        "at",
        "of",
        "for",
        "and",
        "or",
        "with",
        "by",
        "is",
        "be",
        "it",
        "them",
        "then",
        "please",
        "all",
        "up",
        "down",
        "put",
        "place",
        "pick",
        "set",
        "move",
        "other",
        "another",
        "second",
        "next",
        "its",
        "their",
    }
)

# Prepositions whose ORDER carries the task semantics. Any instruction
# containing one of these is refused a Jaccard match unless the ordered
# content tokens also agree -- "stack the blue block on the gray block"
# and "stack the gray block on the blue block" have identical token SETS
# (jaccard 1.0) and opposite meanings.
_POSITIONAL_PREPOSITIONS = frozenset(
    {"on", "in", "into", "onto", "under", "above", "below", "beside", "behind"}
)

# Score keys that carry per-run bookkeeping rather than plan semantics. They
# are stripped before hashing so a plan served from cache re-hashes to the
# entry it came from:
#   - label_corrections: popped by pipeline_execution.execute() before the
#     score is handed to the executor, so the score written back after a run
#     differs from the one served.
#   - __-prefixed keys (__raw_yaml, __planner): planner provenance; the raw
#     YAML text differs in whitespace for byte-identical trees.
#   - verify: the success PREDICATE, not the plan. Two runs that drive the
#     same motion are the same tree whether or not the planner also said how
#     to check the result, so a predicate must never split a cache entry.
_NONSEMANTIC_KEYS = ("label_corrections", "verify")

# Params whose documented default is a no-op. A plan that spells out the
# default must hash identically to the cached tree that omits it, or every
# cached entry misses when the planner emits the new schema.
# Reference semantics: planning/spark_planner.normalize_plan_for_hash.
_DEFAULT_PARAMS = {"grasp_strategy": "auto", "grasp_yaw_deg": None}

_STOPWORDS = frozenset(
    {
        "the",
        "a",
        "an",
        "to",
        "from",
        "in",
        "on",
        "at",
        "of",
        "for",
        "and",
        "or",
        "with",
        "by",
        "is",
        "be",
        "this",
        "that",
        "it",
        "into",
        "onto",
    }
)


def _norm(text: Any) -> str:
    """
    Canonical form of an instruction string for equality comparison.

    Lowercase, underscores and punctuation to spaces, whitespace collapsed.
    The single normalisation dialect in the codebase, exported under the
    public name :func:`normalize_instruction` (bottom of this module); every
    other module calls that one function. ``_norm`` is the in-module name.
    """
    cleaned = re.sub(r"[^\w\s]", " ", str(text).replace("_", " ").lower())
    return " ".join(cleaned.split())


def _tokens(text: str) -> set:
    """
    Lowercase, strip punctuation, drop stopwords.
    """
    return {w for w in _norm(text).split() if w and w not in _STOPWORDS}


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _singular(word: str) -> str:
    """
    Crude, deliberately conservative de-pluralisation.

    Only strips a trailing "s" from words longer than three characters that
    do not end in a doubled or ambiguous sibilant, so "blocks"->"block" and
    "spoons"->"spoon" while "glass", "status" and "this" are left alone.
    Good enough to make "put the spoons in the tray" and "put the spoon in
    the tray" one cache key; it is not a stemmer and must not become one.
    """
    w = str(word)
    if len(w) <= 3 or not w.endswith("s"):
        return w
    if w.endswith(("ss", "us", "is", "es")):
        return w[:-2] if w.endswith("es") and len(w) > 4 else w
    return w[:-1]


def count_differs(query: str, texts) -> bool:
    """True when `query` says "spoons" where every one of `texts` says "spoon".

    The near-exact tiers fold plurals, but a one-spoon contract or tree must not serve a
    request for several.
    """
    words = {w for t in texts for w in _norm(t).split()}
    return any(
        w not in words and _singular(w) != w and _singular(w) in words
        for w in _norm(query).split()
    )


def _canon_word(word: str) -> str:
    """
    Fold one token to its canonical spelling: regional variants first, then
    de-pluralisation, then variants again (so "cubes"->"cube"->"block").
    """
    w = _SPELLING_VARIANTS.get(word, word)
    w = _singular(w)
    return _SPELLING_VARIANTS.get(w, w)


def _near_exact_key(text: str) -> str:
    """
    Order-PRESERVING canonical key for the near-exact retrieval tier.

    Drops filler that never changes which objects a task touches, folds
    spelling variants and plurals, and joins what is left *in order*.
    Order matters: "stack the blue block on the gray block" and "stack the
    gray block on the blue block" have identical token sets but opposite
    meanings, so a set-based key would merge two distinct tasks.

        "put the knife in the tray"                -> "knife tray"
        "pick up the knife and place it in the tray" -> "knife tray"
        "put the spoons in the tray"               -> "spoon tray"
        "stack the blue block on the gray block"   -> "stack blue block gray block"
        "stack the gray block on the blue block"   -> "stack gray block blue block"
    """
    words = [_canon_word(w) for w in _norm(text).split() if w]
    kept = [w for w in words if w and w not in _NEAR_EXACT_FILLER]
    return " ".join(kept)


def _ordered_tokens(text: str) -> List[str]:
    """
    Content tokens in source order, canonicalised. Used to reject a Jaccard
    match whose token multiset agrees but whose ORDER does not.
    """
    return [_canon_word(w) for w in _norm(text).split() if w and w not in _STOPWORDS]


def _is_order_sensitive(text: str) -> bool:
    """
    True when the instruction contains a preposition whose position carries
    the task semantics (A on B vs B on A).
    """
    return bool(set(_norm(text).split()) & _POSITIONAL_PREPOSITIONS)


def _strip_nonsemantic(score: Any) -> Any:
    """
    Deep-copy ``score`` with per-run bookkeeping keys removed.

    Applied at every level, not just the root, so a nested correction map
    cannot shift the hash either.
    """
    if isinstance(score, dict):
        return {
            k: _strip_nonsemantic(v)
            for k, v in score.items()
            if k not in _NONSEMANTIC_KEYS
            and not str(k).startswith("__")
            and not (k in _DEFAULT_PARAMS and v == _DEFAULT_PARAMS[k])
        }
    if isinstance(score, list):
        return [_strip_nonsemantic(v) for v in score]
    return score


def _canonical_score_str(score: Any) -> str:
    """
    Stable string repr of a score for hashing, key order insensitive and
    independent of per-run bookkeeping keys.
    """
    return json.dumps(_strip_nonsemantic(score), sort_keys=True, separators=(",", ":"))


def _hash_entry(instruction: str, score: Any) -> str:
    h = hashlib.blake2b(digest_size=6)
    h.update(_norm(instruction).encode("utf-8"))
    h.update(b"\x00")
    h.update(_canonical_score_str(score).encode("utf-8"))
    return h.hexdigest()


# Public names for the instruction-matching dialect. Every stage that keys off
# the instruction string must agree with the BT cache on "the same
# instruction", or a task can hit its cached tree and miss its SAM3 prompts.
# The underscore-prefixed originals are in-module short names, not the API.
#
#   normalize_instruction -- tier 2 key (exact equality after lowering,
#                            underscores/punctuation to spaces, whitespace
#                            collapsed).
#   near_exact_key        -- tier 3 key (spelling variants, plurals and filler
#                            folded, content-word ORDER preserved, so
#                            "A on B" and "B on A" stay distinct).
normalize_instruction = _norm

near_exact_key = _near_exact_key


@dataclass
class BTEntry:
    hash: str
    instruction: str
    score: Any
    objects: List[str] = field(default_factory=list)
    success: int = 0
    fail: int = 0
    # Runs that executed this tree but produced no verdict (all cameras
    # abstained, occluded, no predicate derivable). Deliberately NOT folded
    # into fail: a systematically-abstaining verifier must make the cache go
    # cold and visible, not wrong. Surfaced on /scores.
    unverified: int = 0
    promoted: bool = False  # Toggled via /api/bt/promote.
    # Extra instruction spellings that resolve to this entry by exact match.
    aliases: List[str] = field(default_factory=list)
    # True for entries installed from configs/bt_seeds/*.yaml. Seeds are
    # authored and version-controlled; run-accumulated entries are not.
    seed: bool = False
    # Author-controlled ordering among entries that match equally well.
    # Higher wins. Run-accumulated entries keep 0.
    priority: int = 0
    # Free-text provenance from the seed file, surfaced in the UI.
    notes: str = ""
    # False when a seed's score has never been confirmed on the real rig.
    verified: bool = True

    def keys(self) -> List[str]:
        """
        Every normalised instruction string that resolves to this entry.
        """
        out = [_norm(self.instruction)]
        out.extend(_norm(a) for a in self.aliases or [])
        return [k for k in out if k]

    def near_keys(self) -> List[str]:
        """
        Every near-exact key (spelling/plural/filler folded, order kept)
        that resolves to this entry.
        """
        out = [_near_exact_key(self.instruction)]
        out.extend(_near_exact_key(a) for a in self.aliases or [])
        return [k for k in out if k]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hash": self.hash,
            "instruction": self.instruction,
            "score": self.score,
            "objects": self.objects,
            "success": self.success,
            "fail": self.fail,
            "unverified": self.unverified,
            "promoted": self.promoted,
            "aliases": self.aliases,
            "seed": self.seed,
            "priority": self.priority,
            "notes": self.notes,
            "verified": self.verified,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "BTEntry":
        return cls(
            hash=d["hash"],
            instruction=d["instruction"],
            score=d["score"],
            objects=list(d.get("objects") or []),
            success=int(d.get("success", 0)),
            fail=int(d.get("fail", 0)),
            unverified=int(d.get("unverified", 0)),
            promoted=bool(d.get("promoted", False)),
            aliases=list(d.get("aliases") or []),
            seed=bool(d.get("seed", False)),
            priority=int(d.get("priority", 0)),
            notes=str(d.get("notes") or ""),
            verified=bool(d.get("verified", True)),
        )


@dataclass
class BTMatch:
    """
    Result of a successful ``BTLibrary.lookup``.
    """

    entry: BTEntry
    source: str  # pin | exact | alias | normalized | similar | seed
    similarity: float

    @property
    def score(self) -> Any:
        """
        Detached copy, so a downstream ``pop()`` cannot mutate the library.
        """
        return copy.deepcopy(self.entry.score)

    @property
    def hash(self) -> str:
        return self.entry.hash


class BTCacheMiss(LookupError):
    """
    Raised by ``resolve`` when no cached BT covers the instruction and
    strict offline mode (``bt.require_cached`` / SPARK_REQUIRE_CACHED_BT)
    forbids falling through to the LLM.
    """
