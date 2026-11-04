"""Detection gate: second-opinion proposer mode, fusion settings and re-detect."""

import logging
import os
from typing import Dict

from spark_core.util import as_bool
from spark_real.config import family_block
from spark_real.perception.detection_fusion import (
    DetectionGate,
    FusionConfig,
    box_iou,
    mask_bbox,
)
from spark_real.perception_settings import (
    _NULL_BACKENDS,
    _PROPOSER_OFF_LABEL,
    PROPOSER_MODE_ALWAYS,
    PROPOSER_MODE_AUTO,
    PROPOSER_MODE_OFF,
    PROPOSER_MODES,
    normalize_proposer_mode,
)
from spark_real.routes import state as _routes_state

logger = logging.getLogger(__name__)


class FusionMixin:
    """Detection gate: second-opinion proposer mode, fusion settings and re-detect."""

    @staticmethod
    def _config_proposer_mode(proposer: Dict) -> str:
        """Which of the three modes a `proposer` block already describes.

        Derived, never stored (the YAML has no `mode:` key). A null/disabled
        backend is `off`, an allowlist is `auto`, no allowlist is `always`.
        """
        backend = str(proposer.get("backend", "null")).lower()
        if backend in _NULL_BACKENDS or not proposer.get("enabled", False):
            return PROPOSER_MODE_OFF
        return PROPOSER_MODE_AUTO if proposer.get("labels") else PROPOSER_MODE_ALWAYS

    def proposer_mode_override(self):
        """`(mode, source)` for a mode somebody forced, else `(None, "config")`.

        Resolution order is YAML -> env -> runtime override, override last.
        `None` = fall back to config.
        """
        for value, source in (
            (getattr(_routes_state, "proposer_mode", None), "runtime"),
            (os.environ.get(self.PROPOSER_MODE_ENV), "env"),
        ):
            try:
                mode = normalize_proposer_mode(value)
            except ValueError as exc:
                # A typo in a shell env var must not take detection down.
                logger.warning("[fusion] ignoring %s proposer mode (%s)", source, exc)
                continue
            if mode is not None:
                return mode, source
        return None, "config"

    def _apply_proposer_mode(self, cfg: Dict, mode: str) -> Dict:
        """Rewrite the `proposer` block so `mode` is what the gate is built with."""
        if mode == PROPOSER_MODE_AUTO:
            # By definition the config's own behaviour: touch nothing.
            return cfg
        proposer = dict(cfg.get("proposer") or {})
        if mode == PROPOSER_MODE_OFF:
            proposer["enabled"] = False
            proposer["backend"] = "null"
            proposer["labels"] = [_PROPOSER_OFF_LABEL]
        else:  # PROPOSER_MODE_ALWAYS: pre-allowlist behaviour
            proposer["enabled"] = True
            proposer["labels"] = []
        cfg = dict(cfg)
        cfg["proposer"] = proposer
        return cfg

    def fusion_status(self) -> Dict:
        """Everything the UI shows about the gate. Never builds it.

        `loaded_backend` comes off the cached gate and stays None until a
        detect has actually built one; a status poll must not pull an RF-DETR
        checkpoint onto the GPU.
        """
        settings = self._fusion_settings()
        cfg = FusionConfig.from_dict(settings)
        proposer = dict(settings.get("proposer") or {})
        override, source = self.proposer_mode_override()
        gate = self._fusion_gate or None

        if self._fusion_gate_error is not None:
            gate_state = "error"
        elif not cfg.enabled:
            gate_state = "off"
        elif gate is not None:
            gate_state = "on"
        else:
            gate_state = "unbuilt"

        return {
            "mode": self._config_proposer_mode(proposer),
            "override": override,
            "source": source,
            "modes": list(PROPOSER_MODES),
            # read-only: the whole fusion gate, and what is behind it
            "fusion_enabled": bool(cfg.enabled),
            "gate_state": gate_state,
            "backend": str(proposer.get("backend", "null")).lower(),
            "loaded_backend": getattr(getattr(gate, "proposer", None), "name", None),
            "labels": [lbl for lbl in cfg.proposer_labels() if lbl != _PROPOSER_OFF_LABEL],
            "error": (
                str(self._fusion_gate_error) if self._fusion_gate_error is not None else None
            ),
        }

    def reset_detection_gate(self) -> None:
        """Drop the memoised gate so the next detect() rebuilds it.

        detection_gate() caches on first use (including the cached-OFF and
        cached-ERROR states), so ANY runtime change to the fusion settings that
        does not come through here is inert until the server restarts.
        """
        self._fusion_gate = None
        self._fusion_gate_error = None
        logger.info("[fusion] gate cache cleared; next detect rebuilds it")

    def _fusion_settings(self) -> Dict:
        """Resolve the `perception.fusion` block, env last.

        Same resolution order ScoreExecutor._apply_family_tuning uses:
        profile.raw when the server loaded an overlay, else the packaged
        configs/<family>_default.yaml. `perception.reprompt_min_conf` and
        `reprompt_max_attempts` are this gate's knobs.
        """
        family = (getattr(self.config, "robot_family", "ur10e") or "ur10e").lower()
        per = dict(family_block(getattr(self, "profile", None), family, "perception"))
        cfg = dict((per.get("fusion") or {}))
        if per.get("reprompt_min_conf") is not None:
            cfg.setdefault("low_quality_conf", float(per["reprompt_min_conf"]))
        if per.get("reprompt_max_attempts") is not None:
            cfg.setdefault("reprompt_max_attempts", int(per["reprompt_max_attempts"]))

        env = os.environ.get(self.FUSION_ENV)
        if env is not None:
            cfg["enabled"] = as_bool(env, False)

        # Last, so it wins: the operator's proposer-mode override. It only ever
        # rewrites the `proposer` sub-block, so `enabled` -- the mask-quality
        # gate itself -- is not something this switch can take down.
        mode, _source = self.proposer_mode_override()
        if mode is not None:
            cfg = self._apply_proposer_mode(cfg, mode)
        return cfg

    def detection_gate(self):
        """The live fusion gate, or None when it is switched off.

        Turned on by `perception.fusion.enabled: true` in the family YAML or by
        SPARK_DETECTION_FUSION=1. A backend that was asked for and cannot be
        built raises here and keeps raising.
        """
        if self._fusion_gate_error is not None:
            raise self._fusion_gate_error
        if self._fusion_gate is not None:
            return self._fusion_gate or None

        cfg = FusionConfig.from_dict(self._fusion_settings())
        if not cfg.enabled:
            self._fusion_gate = False
            return None
        try:
            gate = DetectionGate(cfg)
        except Exception as exc:  # noqa: BLE001 - re-raised, never swallowed
            self._fusion_gate_error = RuntimeError(
                f"detection fusion is enabled but its box proposer is unusable: {exc}"
            )
            logger.error("[fusion] %s", self._fusion_gate_error)
            raise self._fusion_gate_error from exc

        self._fusion_gate = gate
        logger.info(
            "[fusion] gate ON: proposer=%s, low_quality_conf=%.2f, "
            "reprompt_max_attempts=%d, write_confidence=%s",
            getattr(gate.proposer, "name", "?"),
            cfg.low_quality_conf,
            cfg.reprompt_max_attempts,
            cfg.write_confidence,
        )
        return gate

    def _fusion_redetect(self, gate, rgb, depth, cal, secondary_score):
        """Build the ASPIRE `redetect(prompt, box)` callback for one camera.

        The alternate prompt is a real SAM3 pass through _detect_camera, so a
        reprompt result carries a real mask and a real backprojected position.
        A `box` (a second detector's partner box) is honoured by asking for
        every instance of the prompt and keeping the one that overlaps the box.
        """

        def redetect(prompt, box=None):
            multi = {prompt} if box is not None else set()
            cands = self._detect_camera(rgb, depth, cal, [prompt], multi, secondary_score)
            if not cands:
                return None
            if box is not None:
                overlapping = [
                    c
                    for c in cands
                    if box_iou(mask_bbox(c.mask) or c.bbox or (0, 0, 0, 0), box)
                    >= gate.cfg.assoc_iou
                ]
                cands = overlapping or []
            if not cands:
                return None
            return max(cands, key=lambda c: float(c.confidence or 0.0))

        return redetect

    def _apply_fusion(self, gate, detections, rgb, depth, cal, cam_name, secondary_score) -> None:
        """Fuse + ASPIRE-repair one camera's detections, in place."""
        try:
            results = gate.apply(
                detections,
                rgb=rgb,
                redetect=self._fusion_redetect(gate, rgb, depth, cal, secondary_score),
            )
        except Exception as exc:  # noqa: BLE001 - the gate must never stop perception
            logger.warning(
                "[fusion] gate pass failed on %s (%s); detections unchanged",
                cam_name,
                exc,
            )
            return
        for res in results:
            logger.info("[fusion] %s %s", cam_name, res.describe())
