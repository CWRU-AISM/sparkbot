"""SPARK task planner: instruction + annotated views + detections -> a SPARK score (YAML behavior tree).

The system prompt is the frozen text in prompts/system_<family>.txt that every reported number used,
and validate_score checks node types against the frozen prompts/skill_names.txt. Backends: llm.py.
"""

import functools
import json
import logging
import re
from pathlib import Path

import yaml
from PIL import Image as _PILImage

from spark_core import bt_grammar
from spark_core.llm import LLMClient
from spark_core.plan_schema import (
    _repair_yaml_text,
    parse_plan_yaml,
    sanitize_score,
    validate_plan_extensions,
)

logger = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


@functools.lru_cache(maxsize=None)
def load_prompt(name: str) -> str:
    return (PROMPTS_DIR / name).read_text(encoding="utf-8")


PLANNER_IMAGE_MAX_EDGE = 1024


def _downscale_for_planner(img):
    """Shrink a view before it goes to the planner.

    Two full-size 1280x720 frames push a plan call past the ~120 s API
    deadline (504 DEADLINE_EXCEEDED). The planner reads labels, layout and
    coarse orientation off these frames; every number it acts on arrives as
    text in the detection list, so resolution beyond ~1k on the long edge
    buys nothing.
    """
    try:
        w, h = img.size
        long_edge = max(w, h)
        if long_edge <= PLANNER_IMAGE_MAX_EDGE:
            return img
        scale = PLANNER_IMAGE_MAX_EDGE / float(long_edge)
        return img.resize(
            (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
            _PILImage.LANCZOS,
        )
    except Exception:  # noqa: BLE001 - never let a resize stop a plan
        return img


class SPARKPlanner(LLMClient):
    """Generates SPARK scores from language + visual input."""

    def __init__(self, llm_backend=None, api_key=None, model=None, robot_family="ur10e", temperature=0.3):
        self.robot_family = (robot_family or "ur10e").lower()
        if not (PROMPTS_DIR / f"system_{self.robot_family}.txt").exists():
            raise ValueError(f"no frozen planner prompt for robot family {self.robot_family!r}")
        super().__init__(llm_backend=llm_backend, api_key=api_key, model=model, temperature=temperature)

    def _build_system_prompt(self) -> str:
        return load_prompt(f"system_{self.robot_family}.txt")

    def _generate_phase1(self, contents) -> str:
        return self.generate(contents, tag="prompt_generation", model=self.PHASE1_MODEL)

    def assess_completion(self, instruction, images, evidence=""):
        """Adjudicate a disputed run: is the task ALREADY done in these images?

        The terminal recovery rung calls this BEFORE replanning, on CLEAN
        (unannotated) frames, because overlay paint sits on the objects whose
        state is in dispute. It is a second opinion against verifier false
        negatives.

        ``images`` is a list of (camera_name, HxWx3 rgb). Returns
        {"complete": bool, "confidence": float 0..1, "reason": str};
        on any error returns complete=False, confidence=0.0 (never blocks
        the replan path).
        """
        try:
            prompt = (
                "You are inspecting a robot workspace AFTER a manipulation "
                "attempt. Decide whether the task is ALREADY COMPLETE as "
                "photographed.\n"
                f"Task: {instruction}\n"
                + (f"The automated verifier reported: {evidence}\n" if evidence else "")
                + "Automated verifiers here have known FALSE NEGATIVES (a "
                "correctly seated tool can be reported as a failure), so "
                "judge only from the images. Answer with STRICT JSON: "
                '{"complete": true|false, "confidence": 0.0-1.0, '
                '"reason": "<one sentence naming the visual evidence>"}. '
                "Set complete=true only when the images clearly show the "
                "goal state; when unsure, complete=false with low confidence."
            )
            contents = [prompt]
            for name, img in images or []:
                contents.append(f"Camera '{name}':")
                pil = (
                    img if isinstance(img, _PILImage.Image)
                    else _PILImage.fromarray(img)
                )
                # Same 1024-edge cap as every other planner call: byte size,
                # not image count (3,600/request allowed), is what exceeds the
                # deadline.
                contents.append(_downscale_for_planner(pil))
            text = self._generate_phase1(contents)
            match = re.search(r"\{.*\}", text, re.DOTALL)
            out = json.loads(match.group()) if match else {}
            return {
                "complete": bool(out.get("complete", False)),
                "confidence": float(out.get("confidence", 0.0) or 0.0),
                "reason": str(out.get("reason", ""))[:300],
            }
        except Exception as exc:  # noqa: BLE001 - adjudication is best-effort
            logger.warning("[adjudicate] assess_completion failed: %s", exc)
            return {"complete": False, "confidence": 0.0, "reason": f"error: {exc}"}

    def generate_prompts(self, instruction: str, scene_image=None) -> list:
        """
        Phase 1: Ask the LLM to look at the scene and decide what to detect.

        For simple instructions like "pick up the red block", the regex
        extractor works fine.  This method is for complex/ambiguous tasks
        where the LLM needs to see the scene to decide what objects to find.

        Returns:
            List of prompt strings for SAM3 detection.
        """
        context = f"Instruction: {instruction}\n"

        if scene_image is not None:
            contents = [
                load_prompt("prompt_generation.txt") + "\n\n" + context,
                (
                    scene_image
                    if isinstance(scene_image, _PILImage.Image)
                    else _PILImage.fromarray(scene_image)
                ),
            ]
        else:
            contents = load_prompt("prompt_generation.txt") + "\n\n" + context

        response_text = self._generate_phase1(contents)

        # Parse the JSON list from the response. Open models often wrap each phrase in an object
        # ({"prompt": "black bowl"}); take the first string value of each.
        match = re.search(r"\[.*\]", response_text, re.DOTALL)
        if match:
            try:
                items = json.loads(match.group())
                if isinstance(items, list):
                    prompts = []
                    for it in items:
                        if isinstance(it, dict):
                            it = next((v for v in it.values() if isinstance(v, str)), None)
                        if isinstance(it, str) and it.strip():
                            prompts.append(it.strip())
                    if prompts:
                        return prompts
            except json.JSONDecodeError:
                pass

        # Fallback: split by commas/newlines, dropping JSON punctuation
        return [
            p for p in (re.sub(r'[\[\]{}"]|\bprompt\s*:', "", x).strip().strip("'") for x in re.split(r"[,\n]", response_text))
            if p
        ]

    def propose_alt_prompts(
        self,
        target_label: str,
        tried_prompts,
        scene_image=None,
        max_prompts: int = 4,
    ) -> list:
        """
        Ask the LLM to name an object SAM3 could not find, given the frame.

        Last-resort rung of the perception escalation in
        ``PerceptionMixin.detect_for_task``, reached after the task's
        registered prompt and its ``alt_prompts`` both came back empty.

        Same key rotation and retry policy as :meth:`generate_prompts`.
        Returns a ranked list of new prompt strings, or ``[]``; the caller
        treats ``[]`` and an exception identically and falls through to
        ``fallback.on_mismatch``.
        """
        tried = [str(p) for p in (tried_prompts or []) if str(p).strip()]
        context = (
            f"Object to find: {target_label}\n"
            f"Prompts already tried and rejected: {json.dumps(tried)}\n"
            f"Return at most {int(max_prompts)} alternatives.\n"
        )
        if scene_image is not None:
            contents = [
                load_prompt("alt_prompt_proposal.txt") + "\n\n" + context,
                (
                    scene_image
                    if isinstance(scene_image, _PILImage.Image)
                    else _PILImage.fromarray(scene_image)
                ),
            ]
        else:
            contents = load_prompt("alt_prompt_proposal.txt") + "\n\n" + context

        response_text = self._generate_phase1(contents)

        match = re.search(r"\[.*?\]", response_text or "", re.DOTALL)
        if not match:
            return []
        try:
            proposals = json.loads(match.group())
        except json.JSONDecodeError:
            return []
        if not isinstance(proposals, list):
            return []
        # The caller sanitises again (it must, since it also handles a raising
        # or absent planner), but a model that answers with numbers or dicts
        # should not travel any further than this.
        return [str(p).strip() for p in proposals if isinstance(p, str) and p.strip()]

    # Geometry fields rendered into the planner context when present. Absent
    # keys are skipped so an old caller's detection_details still works.
    _GEOMETRY_FIELDS = (
        ("aspect_ratio", "aspect_ratio={:.2f}"),
        ("obb_minor_mm", "obb_minor={:.0f}mm"),
        ("obb_angle_deg", "obb_angle={:.0f}deg"),
        ("obb_confidence", "obb_confidence={:.2f}"),
        # Mask-derived twin of the two above. Prefer these when the OBB is
        # flagged untrustworthy: they come from the silhouette, not the depth.
        ("mask_angle_deg", "mask_angle={:.0f}deg"),
        ("mask_aspect_ratio", "mask_ar={:.2f}"),
        ("container_depth_cm", "container_depth={:.1f}cm"),
        ("mask_area_frac", "mask_area={:.3%}"),
    )

    @classmethod
    def _format_geometry(cls, detail: dict) -> str:
        # ", key=value" fragments for the geometry fields present on a detection.
        parts = []
        for key, fmt in cls._GEOMETRY_FIELDS:
            value = detail.get(key)
            if value is not None:
                parts.append(fmt.format(value))
        if detail.get("camera"):
            parts.append(f"cam={detail['camera']}")
        if detail.get("low_quality"):
            parts.append("LOW_QUALITY_MASK")
        if detail.get("depth_stretched_the_obb"):
            parts.append("OBB_ANGLE_UNRELIABLE(use mask_angle)")
        if detail.get("mask_is_round"):
            parts.append("NO_MAJOR_AXIS")
        if detail.get("reprompt_attempts"):
            parts.append(f"reprompts={detail['reprompt_attempts']}")
        return (", " + ", ".join(parts)) if parts else ""

    def generate_score(
        self,
        instruction: str,
        annotated_image=None,
        keypoint_labels: list = None,
        detection_details: list = None,
        fewshot_examples: list = None,
        mask_overlay_image=None,  # accepted for bench-caller API parity; ignored
        wrist_image=None,         # accepted for bench-caller API parity; ignored
        temperature: float = None,
        scene_camera: str = None,
    ) -> dict:
        """
        Generate a SPARK score from instruction and visual context.

        Args:
            instruction: Natural language task description
            annotated_image: PIL Image with keypoint annotations (optional)
            keypoint_labels: List of detected keypoint labels

        Returns:
            Parsed YAML score as dict
        """
        # Per-call temperature override: the bench's diversity-sampling
        # path (shadow best-of-N) requests temp > 0 per sample. Callers
        # there construct a fresh planner per call, so mutating the
        # instance default is contained.
        if temperature is not None:
            self.temperature = float(temperature)

        context = f"Instruction: {instruction}\n"
        if keypoint_labels:
            context += f"Available keypoints: {', '.join(keypoint_labels)}\n"
        if detection_details:
            context += (
                "Detection details (use these to judge which detections are real):\n"
            )
            for dd in detection_details:
                context += f"  - {dd['label']}: confidence={dd['confidence']:.0%}"
                if dd.get("position_3d"):
                    p = dd["position_3d"]
                    context += f", pos=({p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f})"
                context += self._format_geometry(dd)
                context += "\n"
            context += (
                "Read confidence and obb_confidence together with the image: a "
                "high aspect_ratio on a low-confidence or low-obb_confidence "
                "mask is noise, not an elongated object.\n"
            )
            context += "IMPORTANT: Verify each detection label against what you see in the image. If a label is CLEARLY wrong (e.g., 'knife handle 2' is actually a spoon), add a 'label_corrections' field to the YAML mapping the wrong label to the correct one. Use the CORRECTED label in your plan's keypoint_label fields. Example:\n"
            context += "label_corrections:\n  knife handle 2: spoon handle\n"
            context += "Only correct labels when the object type is clearly wrong. Do NOT change color descriptions (e.g., don't rename 'yellow block' to 'orange block'), color perception varies and the original detection label is what the robot uses for tracking.\n"
            context += "Ignore detections that don't correspond to any real object in the image.\n"

        # BTLibrary few-shot: prior successful real-robot trials
        # (instruction, score) pairs.  Boosts generalisation across
        # similar instructions without retraining.  Wire-up is in
        # pipeline.py; safe to ignore when caller passes nothing.
        if fewshot_examples:
            context += "\nPrior successful plans for similar tasks:\n"
            for i, (ex_instr, ex_score) in enumerate(fewshot_examples, 1):
                context += f"\nExample {i} (task: {ex_instr}):\n"
                context += "```yaml\n"
                context += yaml.safe_dump(ex_score, sort_keys=False).strip()
                context += "\n```\n"
            context += "\nUse these as guides; emit a single new plan.\n"

        system_prompt = self._build_system_prompt()

        if annotated_image is not None:
            # Multimodal: image(s) + text. annotated_image may be ONE image
            # or a list of (camera_name, image) pairs; a second view
            # corrects the foreshortening of elongated objects in a single
            # fixed camera. The text names the images in order so the
            # model can refer to each.
            views = annotated_image
            if not isinstance(views, (list, tuple)):
                views = [(scene_camera or "scene", views)]
            elif views and not isinstance(views[0], (list, tuple)):
                views = [(scene_camera or "scene", v) for v in views]

            header = system_prompt + "\n\n" + context
            if len(views) > 1:
                header += (
                    "\n\nYou are given " + str(len(views)) + " camera views of "
                    "the SAME scene, in this order: "
                    + ", ".join(str(n) for n, _ in views)
                    + ". They show one scene from different angles, not "
                    "different scenes. Cross-check between them: an object "
                    "that looks round or short in one view is often "
                    "foreshortened there, and the other view shows its true "
                    "extent. Where they disagree, prefer the measured "
                    "geometry in the detection list over either picture.\n"
                )
            contents = [header]
            for _name, _img in views:
                if _img is None:
                    continue
                _p = (
                    _img
                    if isinstance(_img, _PILImage.Image)
                    else _PILImage.fromarray(_img)
                )
                contents.append(_downscale_for_planner(_p))
        else:
            contents = system_prompt + "\n\n" + context
        yaml_text = self._extract_yaml(self.generate(contents, tag="bt_generation"))
        return self.parse_response(yaml_text, keypoint_labels)

    @staticmethod
    def parse_response(yaml_text: str, keypoint_labels: list = None) -> dict:
        """Parse + schema-check one raw LLM response. No network."""
        score = parse_plan_yaml(yaml_text)
        clean, _issues = sanitize_score(score, keypoint_labels)
        return clean

    def _extract_yaml(self, text: str) -> str:
        # Extract YAML from LLM response (may be wrapped in markdown code block).
        if "```yaml" in text:
            start = text.index("```yaml") + 7
            end = text.index("```", start)
            return text[start:end].strip()
        if "```" in text:
            start = text.index("```") + 3
            end = text.index("```", start)
            return text[start:end].strip()
        # No code block: drop the prose preamble, starting at `task:` when
        # the model emitted one.
        return _repair_yaml_text(text)

    def validate_score(self, score: dict, keypoint_labels: list = None) -> list:
        """
        Validate a SPARK score for correctness.

        Returns list of issues (empty if valid). Covers the optional
        extension fields too; an old-style plan adds no issues.
        """
        issues = []
        if "tree" not in score:
            issues.append("Missing 'tree' key")
            return issues

        tree = score["tree"]
        if tree.get("type") not in {"sequence"} | bt_grammar.CONTROL_FLOW_TYPES:
            issues.append("Root tree type should be 'sequence'")

        # Leaf types: the frozen skill set the prompt advertises (registered skills plus executor
        # built-ins). Structural types: sequence, the bt_grammar control flow, and parallel /
        # sync_barrier, which the single-arm executor flattens to sequential.
        valid_types = (
            set(load_prompt("skill_names.txt").split())
            | {"sequence", "parallel", "sync_barrier"}
            | set(bt_grammar.CONTROL_FLOW_TYPES)
        )

        # Recurse one level so the parallel children are also checked.
        def _scan(node, path=""):
            if not isinstance(node, dict):
                return
            t = node.get("type")
            if t is not None and t not in valid_types:
                issues.append(f"{path or 'root'}: unknown type '{t}'")
            for i, child in enumerate(node.get("children", []) or []):
                _scan(child, f"{path}/{i}" if path else f"child[{i}]")

        for i, child in enumerate(tree.get("children", []) or []):
            _scan(child, f"child[{i}]")

        issues.extend(validate_plan_extensions(score, keypoint_labels))
        return issues
