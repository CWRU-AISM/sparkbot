"""The planner prompt every reported number used, frozen (risk R1).

The UR10e system prompt is built from the auto-registered skills, so deleting a skill module would
silently change what the planner sees. The hashes below were recorded on reconcile-jul07 (0fd07d2;
the v4 runner snapshot c3aa15f gives the same prompt). skill_names.txt is the sorted union of the
registered skills and the executor built-ins, one per line. The two phase-1 templates were frozen
from the same source in WP1.
"""
import hashlib
from pathlib import Path

PROMPTS = Path(__file__).resolve().parents[1] / 'prompts'
SYSTEM_SHA256 = 'eb00994124170a9179be76ec12f87fa2317156aead26d868f7b9cadb88d51eab'
SKILLS_SHA256 = '320951b0d58864a5185e4ccda3514028b56543fbf2f92b06d3edfdf0b3e6b916'
# Franka and bimanual prompts: the same builder at reconcile-jul07 29c620b
# (which still reproduces SYSTEM_SHA256 for ur10e). The bimanual text differs
# from that build in one word: the site name before "cloth offsets" reads "Rig"
# (site name removed for the release).
FAMILY_SHA256 = {
    'system_franka.txt': '1a0253b587e9f41b66b007c562796f0ed13c4049ab0de79d16c0ee166edf3059',
    'system_bimanual_franka.txt': 'bffe6adca48ede1c01df869897ca35408348ae7fd9b07aed6b117f6f97872a1e',
}
PHASE1_SHA256 = {
    'prompt_generation.txt': '743fa4b0d82e6d07e3753f0d2a1d3babe4b47a840a2cfdc7690aa0d34d2af598',
    'alt_prompt_proposal.txt': 'bd361a8baa035d8653626559274ff03d1aed8ddee5cb698885d7d7cc925aca2d',
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_frozen_files_match_recorded_hashes():
    assert _sha(PROMPTS / 'system_ur10e.txt') == SYSTEM_SHA256
    assert _sha(PROMPTS / 'skill_names.txt') == SKILLS_SHA256
    for name, sha in {**PHASE1_SHA256, **FAMILY_SHA256}.items():
        assert _sha(PROMPTS / name) == sha, name


def test_planner_builds_the_frozen_prompt():
    from spark_core.planner import SPARKPlanner
    planner = SPARKPlanner.__new__(SPARKPlanner)
    planner.robot_family = 'ur10e'
    assert planner._build_system_prompt() == (PROMPTS / 'system_ur10e.txt').read_text()


def test_prompt_advertises_the_enforced_recovery_caps():
    from spark_core import bt_grammar
    text = (PROMPTS / 'system_ur10e.txt').read_text()
    assert f'Range 1..{bt_grammar.MAX_RETRY_ATTEMPTS};' in text
    assert f'Max {bt_grammar.MAX_SELECTOR_BRANCHES}\n  branches' in text
    assert f'{bt_grammar.MAX_RECOVERY_ATTEMPTS_PER_SCORE} recovery re-attempts' in text
