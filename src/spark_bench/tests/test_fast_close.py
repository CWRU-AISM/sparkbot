"""SPARK_FAST_CLOSE: a close ends once the aperture settles, not before min_n samples or while still moving."""
import numpy as np

from spark_bench.libero_pro.telemetry import aperture_settled, classify_grasp_outcome


def test_aperture_settled():
    closing = list(np.linspace(0.08, 0.04, 40)) + [0.04] * 30  # contact at 40, flat after
    assert not aperture_settled(closing[:59])          # under min_n
    assert aperture_settled(closing)                   # 70 samples, last 30 flat
    assert not aperture_settled(list(np.linspace(0.08, 0.0, 80)))  # still moving
    # Cutting at the settle point keeps the SECURED verdict of the full 300-sample trace.
    full = closing + [0.04] * 230
    assert classify_grasp_outcome(closing).outcome == classify_grasp_outcome(full).outcome
