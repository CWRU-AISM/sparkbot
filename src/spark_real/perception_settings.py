"""Second-opinion proposer modes and shared perception constants."""



# Operator-facing modes of the second-opinion box proposer, a view of the
# `perception.fusion.proposer` block: `auto` is whatever that block says,
# `always` drops its allowlist, `off` swaps in the null backend. The master
# `perception.fusion.enabled` switch is untouched by all three, so the
# mask-quality channel (which vetoed a bogus 69 deg wrist rotation off a
# meaningless plushie OBB) stays up even in `off`.
PROPOSER_MODE_AUTO = "auto"

PROPOSER_MODE_ALWAYS = "always"

PROPOSER_MODE_OFF = "off"

PROPOSER_MODES = (PROPOSER_MODE_AUTO, PROPOSER_MODE_ALWAYS, PROPOSER_MODE_OFF)

# Backend names box_proposals.build_proposer treats as "no detector".
_NULL_BACKENDS = ("null", "off", "none", "")

# An allowlist entry no real label can canonicalise to, so `off` holds even if
# a backend gets built anyway (SPARK_BOX_PROPOSER bypasses the backend key
# inside build_proposer): proposer_trusts() is False for every label and every
# detection takes the ABSTAIN path, confidence bit-identical.
_PROPOSER_OFF_LABEL = "__proposer_off__"


def normalize_proposer_mode(mode):
    """A member of PROPOSER_MODES, or None for "no override".

    Raises ValueError on anything else.
    """
    if mode is None:
        return None
    m = str(mode).strip().lower()
    if not m:
        return None
    if m not in PROPOSER_MODES:
        raise ValueError(f"unknown proposer mode {mode!r}; known: {list(PROPOSER_MODES)}")
    return m


# Capture, SAM3 detection, multi-view merge, and annotation.
# Mixed into the pipeline class.
# Nothing resting on the table stands taller than this. A detection above
# it is a depth failure, not a tall object, and is worth correcting from
# the other camera even on weak evidence.
MAX_ON_TABLE_HEIGHT_M = 0.35
