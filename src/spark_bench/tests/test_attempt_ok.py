"""Recovery gate under SPARK_NO_ORACLE: the robot's own evidence decides, never the simulator predicate."""
from spark_bench.fair import execution


class _Env:
    def check_success(self):
        raise AssertionError('predicate read under SPARK_NO_ORACLE')


def test_attempt_ok_uses_robot_evidence(monkeypatch):
    monkeypatch.setenv('SPARK_NO_ORACLE', '1')
    monkeypatch.delenv('SPARK_ORACLE_RETRY', raising=False)
    env = _Env()
    assert execution._attempt_ok(env, {}) is True                     # no evidence of failure
    assert execution._attempt_ok(env, {'grasp_outcomes': [{'outcome': 'unknown'}]}) is True
    assert execution._attempt_ok(env, {'grasp_outcomes': [{'outcome': 'empty_close'}]}) is False
    tm = {'grasp_outcomes': [{'outcome': 'slip'}, {'outcome': 'secured'}]}
    assert execution._attempt_ok(env, tm, 1) is True                  # only this attempt's grasps count
    tm = {'scene_abort': {'reason': 'target_missing'}}
    assert execution._attempt_ok(env, tm) is False and tm['scene_aborts']  # abort consumed and kept
    assert execution._attempt_ok(env, tm) is True


def test_oracle_retry_reads_only_the_attempt_check(monkeypatch):
    """SPARK_ORACLE_RETRY=1 (Table 1 protocol): the retry decision reads the predicate; other reads stay gated."""
    monkeypatch.setenv('SPARK_NO_ORACLE', '1')
    monkeypatch.setenv('SPARK_ORACLE_RETRY', '1')

    class _Done:
        def check_success(self):
            return True
    assert execution._attempt_ok(_Done(), {'grasp_outcomes': [{'outcome': 'empty_close'}]}) is True
    assert execution._oracle_gate(_Done()) is False


def test_layered_recovery_stops_on_robot_evidence(monkeypatch):
    """Under SPARK_NO_ORACLE a retry whose grasp held ends the ladder (no re-execution on a solved scene)."""
    from types import SimpleNamespace
    monkeypatch.setenv('SPARK_NO_ORACLE', '1')
    monkeypatch.delenv('SPARK_ORACLE_RETRY', raising=False)
    runs = []
    monkeypatch.setattr(execution, '_find_ee_site', lambda m: -1)
    monkeypatch.setattr(execution, '_retract_arm', lambda *a, **k: None)
    monkeypatch.setattr(execution, 'redetect_agentview', lambda *a, **k: SimpleNamespace(det_map={'can': 1}))
    monkeypatch.setattr(execution, 'relation_bind_pick', lambda *a, **k: None)
    monkeypatch.setattr(execution, 'mujoco', SimpleNamespace(mj_forward=lambda *a: None))
    monkeypatch.setattr(execution, 'compute_scene_diff', lambda *a, **k: None)
    monkeypatch.setattr(execution, 'attribute_failure', lambda *a, **k: execution.Layer.EXECUTION)

    def _exec(env, score, det, *a, trial_meta=None, **k):
        runs.append(1)
        trial_meta['grasp_outcomes'].append({'outcome': 'secured'})
    monkeypatch.setattr(execution, '_execute', _exec)
    cfg = SimpleNamespace(privileged=False, max_local_retries=1, verbose=False, no_gemini=True)
    det = SimpleNamespace(det_map={}, dets=[])
    tm = {'grasp_outcomes': [{'outcome': 'empty_close'}]}
    ok = execution._layered_recovery(_Env(), None, None, cfg, None, {}, det, 'pick', [], None, 'can', 'basket',
                                     None, None, None, tm, lambda *a, **k: None)
    assert ok is True and len(runs) == 1
