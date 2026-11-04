"""ProSuite refuses a missing perturbed suite instead of running the unperturbed base suite."""
import pytest

env = pytest.importorskip('spark_bench.fair.env')


def test_missing_perturbed_suite_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(env, 'get_libero_path', lambda kind: str(tmp_path / kind))
    with pytest.raises(FileNotFoundError, match='libero_object_swap'):
        env.ProSuite('libero_object', 'libero_object_swap')


def test_upstream_libero_registers_no_perturbed_suites():
    env.check_upstream_libero('libero_object_swap')   # raises SystemExit under the LIBERO-PRO fork
    with pytest.raises(SystemExit):
        env.check_upstream_libero('libero_object')


def test_perturbation_suite_names():
    assert [env.get_perturbation_suite('libero_goal', p) for p in ('position', 'task', 'language', 'object')] == [
        'libero_goal_swap', 'libero_goal_task', 'libero_goal_lan', 'libero_goal_object']
