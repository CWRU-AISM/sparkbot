"""Spoon direction, place verify and the plural guard: the Nov 5 robot run."""

from types import SimpleNamespace

import numpy as np
import pytest

from spark_real.bt_keys import count_differs
from spark_real.control.executor_verify import VerifyMixin
from spark_real.control.identity_binder import IdentityBinder
from spark_real.planning import tool_ends
from spark_real.perception.prompt_registry import load_registry
from spark_real.skills import place_in_slot as pis


class Ex:
    GRASP_ORIENTATION = [np.pi, 0.0, 0.0]
    MAX_YAW_OFFSET = np.deg2rad(190)

    def __init__(self, head, now_deg):
        from scipy.spatial.transform import Rotation as R

        self._last_pick_label = "tool 1"
        self._tool_heads = {} if head is None else {"tool 1": head}
        self._now = (R.from_euler("z", now_deg, degrees=True) * R.from_rotvec(self.GRASP_ORIENTATION)).as_rotvec()

    def current_orientation(self, fallback=None):
        return list(self._now)

    def _oriented_grasp(self, yaw):
        from scipy.spatial.transform import Rotation as R

        return list((R.from_euler("z", yaw) * R.from_rotvec(self.GRASP_ORIENTATION)).as_rotvec())


@pytest.mark.parametrize("grasp_deg", [-90, -34, 10, 93])
@pytest.mark.parametrize("head_rel", [0, 180])
def test_heads_end_up_the_same_way_whatever_the_grasp(grasp_deg, head_rel):
    slot = np.deg2rad(78)
    ang = np.deg2rad(grasp_deg + head_rel)
    ex = Ex(np.array([np.cos(ang), np.sin(ang)]), grasp_deg)
    yaw, _ = pis._point_head_along_slot(ex, slot, ex._oriented_grasp(slot))
    final = ang + yaw - np.deg2rad(grasp_deg)
    assert np.cos(final - slot) > 0.99


def test_flip_past_the_wrist_limit_keeps_the_slot_axis():
    ex = Ex(np.array([-1.0, 0.0]), 0)
    ex.MAX_YAW_OFFSET = np.deg2rad(100)
    slot = np.deg2rad(78)
    yaw, _ = pis._point_head_along_slot(ex, slot, ex._oriented_grasp(slot))
    assert yaw == slot


def test_unknown_head_passes_the_yaw_through():
    ex = Ex(None, 0)
    assert pis._point_head_along_slot(ex, 0.5, [1, 2, 3]) == (0.5, [1, 2, 3])


def _bar(label="tool 1", xr=(150, 490), yr=(230, 250)):
    # Camera straight down at z=1: image x -> world x, image y -> world -y.
    K = np.array([[500.0, 0, 320], [0, 500.0, 240], [0, 0, 1]])
    T = np.eye(4)
    T[:3, :3] = np.diag([1.0, -1.0, -1.0])
    T[2, 3] = 1.0
    mask = np.zeros((480, 640), np.uint8)
    mask[yr[0]:yr[1], xr[0]:xr[1]] = 1
    det = SimpleNamespace(label=label, mask=mask, camera="birdview", position_3d=[0.0, 0.0, 0.0])
    return det, {"birdview": {"calibration": SimpleNamespace(extrinsic=T, intrinsic_matrix=K)}}


def test_tool_ends_turn_the_planners_letter_into_a_world_direction():
    det, captures = _bar()
    ends = tool_ends.measure([det], captures)
    assert ends["tool 1"]["A"][1][0] > ends["tool 1"]["B"][1][0]  # A is the larger world x
    assert tool_ends.headings(ends, {"tool 1": "A"})["tool 1"][0] > 0.99
    assert tool_ends.headings(ends, {"tool 1": "b"})["tool 1"][0] < -0.99
    assert tool_ends.headings(ends, {"tool 1": None}) == {}
    assert tool_ends.headings(ends, "nonsense") == {}


def test_only_elongated_masks_get_ends():
    det, captures = _bar("tray", (200, 440), (150, 350))
    assert tool_ends.measure([det], captures) == {}


def test_ends_are_only_drawn_on_their_own_camera():
    det, captures = _bar()
    ends = tool_ends.measure([det], captures)
    img = np.zeros((480, 640, 3), np.uint8)
    views, marked = tool_ends.draw([("sideview", img), ("birdview", img)], ends)
    assert marked == {"tool 1"} and views[0][1].sum() == 0 and views[1][1].sum() > 0
    assert views[-1][0] == "zoomed tools"
    assert tool_ends.draw([("sideview", img)], ends)[1] == set()


def test_a_spoon_still_on_the_table_is_not_the_placed_one():
    class E:
        detection_map = {
            "spoons 2": {"position_3d": [-0.807, 0.331, -0.27]},
            "spoons 3": {"position_3d": [-0.676, 0.090, -0.27]},
            "tray": {"position_3d": [-0.91, -0.31, -0.26]},
        }
        _placed_labels = set()

    # The false "landed far": spoon 2 was blamed for spoon 3's position.
    assert pis._still_on_table(E, "spoons 2", np.array([-0.676, 0.090]))
    assert not pis._still_on_table(E, "spoons 2", np.array([-0.5, 0.3]))  # a real drop
    E._placed_labels = {"spoons 3"}
    assert not pis._still_on_table(E, "spoons 2", np.array([-0.676, 0.090]))


def test_binder_ignores_siblings_that_also_moved():
    plan = {f"spoons {i}": {"position_3d": [0.1 * i, 0.0, 0.0]} for i in (1, 2, 3)}
    seen = [SimpleNamespace(label="spoons", position_3d=[-0.9, -0.3, -0.25], confidence=0.9)]
    binder = IdentityBinder(plan, seen, manipulated=set(plan))
    assert binder.lookup("spoons 1") is not None


def test_plural_request_is_not_the_one_spoon_contract():
    spoon = ("put the spoon in the tray", "pick up the spoon and place it in the tray")
    assert count_differs("Pick up the spoons and place in the tray", spoon)
    assert not count_differs("place the spoon in the tray", spoon)
    assert not count_differs("put the spoons in the tray", ("put the spoons in the tray",))
    registry = load_registry()
    assert registry.lookup("Pick up the spoons and place in the tray") is None
    assert registry.lookup("pick up the spoon and place it in the tray") is not None


@pytest.mark.parametrize(
    "reply, status",
    [
        ('{"success": true, "reason": "all three are in the tray"}', "pass"),
        ('{"success": false, "reason": "one is on the table"}', "unverified"),
        ("not json", "unverified"),
    ],
)
def test_llm_rung_can_only_upgrade_an_abstain(reply, status):
    planner = SimpleNamespace(generate=lambda *a, **k: reply)
    rgb = np.zeros((4, 4, 3), np.uint8)
    pipeline = SimpleNamespace(_planner=planner, capture=lambda: {"birdview": {"rgb": rgb}})
    ex = SimpleNamespace(_pipeline=pipeline)
    outcome = SimpleNamespace(status="unverified", reason="no 3d pass vote")
    assert VerifyMixin._fold_in_llm_rung(ex, outcome, {"task": "t"}).status == status


def test_llm_rung_asks_about_the_operators_words():
    seen = []
    planner = SimpleNamespace(generate=lambda c, **k: seen.append(c[0]) or '{"success": true}')
    rgb = np.zeros((4, 4, 3), np.uint8)
    pipeline = SimpleNamespace(_planner=planner, capture=lambda: {"birdview": {"rgb": rgb}})
    ex = SimpleNamespace(_pipeline=pipeline, _task_instruction="put spoons in the tray")
    VerifyMixin._fold_in_llm_rung(
        ex, SimpleNamespace(status="unverified", reason=""), {"task": "all in the same slot"}
    )
    assert "put spoons in the tray" in seen[0] and "same slot" not in seen[0]


def test_redetect_keeps_a_slotted_trays_plan_time_shape():
    from spark_real.control import redetect

    slots = [{"slot_idx": 0, "world_xyz": [-0.8, -0.26, -0.26]}]
    tray = {"position_3d": [-0.91, -0.31, -0.26], "aspect_ratio": 1.2, "orientation_angle": 1.45, "slots": slots}
    fresh = SimpleNamespace(position_3d=[-0.917, -0.318, -0.26], orientation_angle=0.3, aspect_ratio=1.6)
    pipeline = SimpleNamespace(capture=lambda: {}, detect=lambda *a, **k: [fresh], merge_detections=lambda d: d)
    ex = SimpleNamespace(_pipeline=pipeline, detection_map={"tray": tray}, profile=None)
    redetect.sticky_settings = lambda e: {"enabled": False}
    redetect.redetect_single(ex, "tray")
    got = ex.detection_map["tray"]
    assert got["position_3d"][0] == -0.917 and got["aspect_ratio"] == 1.2 and got["slots"] == slots


def test_cached_tree_loses_a_scene_yaw_but_keeps_a_place_yaw():
    from spark_real.pipeline_plan import _drop_scene_yaw, _places_in_slot

    grasp = {"type": "grasp", "params": {"grasp_strategy": "obb", "grasp_yaw_deg": 170}}
    place = {"type": "move_to_keypoint", "params": {"keypoint_label": "bed", "grasp_yaw_deg": 90}}
    slot = {"type": "place_in_slot", "params": {"slot_idx": 0}}
    score = {"tree": {"type": "sequence", "children": [grasp, place, slot]}}
    _drop_scene_yaw(score)
    assert "grasp_yaw_deg" not in grasp["params"] and place["params"]["grasp_yaw_deg"] == 90
    assert _places_in_slot(score)
