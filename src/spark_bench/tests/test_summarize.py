"""summarize: trials keyed (suite, pert, task, init) from runner results and campaign queues; macros and CIs."""
import json

from spark_bench import summarize


def _result(suite, pert, task_id, offset, wins, timing=None):
    metas = [{'trial': i, 'success': w, **(timing or {})} for i, w in enumerate(wins)]
    return {'suite': suite, 'num_trials': len(wins), 'trial_offset': offset, 'perturbations': {pert: {
        'task_details': [{'task_id': task_id, 'trial_offset': offset, 'trials': wins, 'trial_meta': metas}]}}}


def test_runner_results_and_campaign_queue(tmp_path):
    (tmp_path / 'a').mkdir()
    (tmp_path / 'a' / 'object.json').write_text(json.dumps(_result('object', 'position', 0, 0, [True, False])))
    # the same trials again (counted once) and an unfinished checkpoint (ignored)
    (tmp_path / 'b').mkdir()
    (tmp_path / 'b' / 'object.json').write_text(json.dumps(_result('object', 'position', 0, 0, [True, False])))
    (tmp_path / 'b' / 'object.partial.json').write_text(json.dumps(_result('object', 'position', 1, 0, [True])))
    # campaign queue: item line gives task and offset; the result file has no task_id (old runner)
    q = tmp_path / 'q'
    (q / 'out' / '0').mkdir(parents=True)
    (q / 'out' / '1').mkdir(parents=True)
    (q / 'done').mkdir()
    (q / 'done' / '0').touch()
    (q / 'items.txt').write_text('object position 1 7 1 wvrb\nobject position 2 0 1 wvrb\n')
    old = {'suite': 'object', 'perturbations': {'position': {'task_details': [
        {'trial_meta': [{'trial': 0, 'success': True}]}]}}}
    (q / 'out' / '0' / 'object.json').write_text(json.dumps(old))
    (q / 'out' / '1' / 'object.json').write_text(json.dumps(old))   # not done: ignored
    trials, dups = summarize.load([tmp_path])
    assert dups == 2 and sorted(trials) == [('object', 'position', 0, 0), ('object', 'position', 0, 1),
                                            ('object', 'position', 1, 7)]


def test_macros_are_means_of_cell_means():
    trials = {}
    for s, p, rates in (('object', 'position', (1, 0)), ('goal', 'task', (1, 1)), ('spatial', 'language', (0, 0))):
        for t, r in enumerate(rates):
            for i in range(4):
                trials[(s, p, t, i)] = {'success': bool(r), 't_wall_s': 10.0 + i}
    out = summarize.report(trials, 0, ci=True, suite_means=True, timing=True, reps=200)
    assert 'macro development (position+task): 75.0' in '\n'.join(out)   # (50 + 100) / 2
    assert any(line.startswith('macro held-out (language+object): 0.0') for line in out)
    assert any(line.startswith('suite means: object 50.0, goal 100.0, spatial 0.0') for line in out)
    assert any('t_wall_s med 11.5' in line for line in out)


def test_cluster_boot_resamples_tasks():
    lo, hi = summarize.cluster_boot([(10, 10), (0, 10)], reps=500)
    assert lo == 0.0 and hi == 1.0   # two tasks: a resample can draw the same task twice
    assert summarize.cluster_boot([(5, 10)] * 4, reps=100) == (0.5, 0.5)
