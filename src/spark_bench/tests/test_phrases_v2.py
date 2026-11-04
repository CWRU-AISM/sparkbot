"""SPARK_PHRASES_V2=1 swaps in the audited SAM3 phrases at import; unset keeps the original table."""
import os
import subprocess
import sys

_PROBE = "from spark_bench.libero_pro.bddl import _OBJ_VISUAL_PROMPTS as P; print(P['cream_cheese'][0])"


def _first_phrase(v2: str) -> str:
    env = dict(os.environ, SPARK_PHRASES_V2=v2)
    return subprocess.run([sys.executable, '-c', _PROBE], env=env, check=True,
                          capture_output=True, text=True).stdout.strip()


def test_phrase_table_switch():
    assert _first_phrase('1') == 'small blue box'
    assert _first_phrase('0') == 'cream cheese box'
