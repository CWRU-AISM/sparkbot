"""FairConfig field contract (critic C17): exactly the fields the headline runner, the no-LLM smoke path and
the LIBERO-Dyn arms (DynConfig subclasses FairConfig) read."""
import dataclasses

KEPT = {
    # runner and protocol
    'suite', 'perturbation', 'num_trials', 'cam_width', 'cam_height', 'no_gemini', 'verbose',
    'output_dir', 'only_task_id', 'trial_offset', 'freshness_timeout_s',
    # perception arms (headline: adaptive + wrist verify + relation bind; language-only control)
    'use_gemini_prompts', 'no_bddl_hints', 'adaptive_prompts', 'adaptive_k', 'language_only',
    'wrist_verify', 'relation_bind',
    # verify-and-adapt arm of the fixed-plan displacement result (LIBERO-Dyn: --event-captures --scene-diff
    # --layered-recovery)
    'pre_close_retarget_min_cm', 'event_captures', 'scene_diff', 'retarget_max_cm', 'scene_diff_moved_cm',
    'scene_diff_self_radius_cm', 'layered_recovery', 'max_local_retries',
}


def test_fair_config_is_the_contract():
    from spark_bench.fair.config import FairConfig
    fields = {f.name for f in dataclasses.fields(FairConfig)}
    assert fields == KEPT, (sorted(KEPT - fields), sorted(fields - KEPT))
