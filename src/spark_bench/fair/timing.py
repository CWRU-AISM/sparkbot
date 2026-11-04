"""Per-trial wall-time accounting for the headline runner: planner (LLM) calls and SAM3 detection.

``install()`` wraps the planner's OpenAI-backend call and the SAM3 entry points (spark_core.sam3,
spark_core.sim_detect) once; ``reset()`` at the start of a trial, ``snapshot()`` at its end. Nested calls
(an entry point that calls another) are timed once, at the outermost level. The Gemini backend is not
wrapped (the headline uses the open model).
"""
import functools
import time

_acc = {'t_plan_s': 0.0, 'n_llm_calls': 0, 't_perception_s': 0.0}
_depth = {'t_plan_s': 0, 't_perception_s': 0}


def reset() -> None:
    _acc.update(t_plan_s=0.0, n_llm_calls=0, t_perception_s=0.0)


def snapshot() -> dict:
    return {k: (round(v, 3) if isinstance(v, float) else v) for k, v in _acc.items()}


def _wrap(cls, name: str, key: str, count: str = '') -> None:
    f = getattr(cls, name, None)
    if f is None or getattr(f, '_timed', False):
        return

    @functools.wraps(f)
    def g(*a, **k):
        _depth[key] += 1
        t = time.perf_counter()
        try:
            return f(*a, **k)
        finally:
            _depth[key] -= 1
            if _depth[key] == 0:
                _acc[key] += time.perf_counter() - t
                if count:
                    _acc[count] += 1
    g._timed = True
    setattr(cls, name, g)


def install() -> None:
    from spark_core import sam3 as sp, sim_detect
    from spark_core.planner import SPARKPlanner
    _wrap(SPARKPlanner, '_openai_generate', 't_plan_s', 'n_llm_calls')
    _wrap(sim_detect, 'detect_rendered', 't_perception_s')
    for name in ('detect', '_detect_with_rendered_depth'):
        _wrap(sp.SPARKPerception, name, 't_perception_s')
    if sp.Sam3Processor is not None:   # the raw processor: adaptive variants and wrist-verify call it directly
        for name in ('set_image', 'set_text_prompt', 'set_point_prompt'):
            _wrap(sp.Sam3Processor, name, 't_perception_s')


if __name__ == '__main__':
    class _P:
        def outer(self):
            time.sleep(0.01)
            return self.inner()

        def inner(self):
            time.sleep(0.01)
            return 1
    _wrap(_P, 'outer', 't_perception_s')
    _wrap(_P, 'inner', 't_perception_s')
    reset(); _P().outer(); s = snapshot()
    assert 0.018 < s['t_perception_s'] < 0.05, s   # nested inner not double counted
    print('ok', s)
