"""LLM backends for the planner: any OpenAI-compatible server (vLLM; the default) or Gemini.

Backend: SPARK_LLM_BACKEND (openai | gemini), which overrides the constructor argument so runners
select it by env. Model: the constructor argument, else SPARK_LLM_MODEL, else SPARK_GEMINI_MODEL
(older alias), else the backend default. Keys come from the environment only: OPENAI_API_KEY (vLLM
ignores it) or GEMINI_API_KEY. OPENAI_BASE_URL points the openai backend at the server.
"""

import base64
import io
import logging
import os

from PIL import Image

try:
    from google import genai as _genai
    from google.genai import types as _genai_types
except ImportError:
    _genai = None
    _genai_types = None

try:
    from openai import OpenAI as _OpenAI
except ImportError:
    _OpenAI = None

logger = logging.getLogger(__name__)

DEFAULT_MODELS = {
    "openai": "MIRALABS/Qwen3.8-27B-W4A16-AutoRound",
    "gemini": "gemini-3.5-flash",
}
# Hard deadline for every Gemini request (ms): a stalled connection must fail the plan, never hang it.
GEMINI_TIMEOUT_MS = int(os.environ.get("SPARK_GEMINI_TIMEOUT_MS", "120000"))

_backend_logged = False


class LLMClient:
    """One LLM call at a time on the active backend, with per-call token usage in `usage_log`."""

    # Phase-1 model on Gemini (prompt generation). An OpenAI-compatible server serves one model, so
    # any gemini-* name collapses to self.model there.
    PHASE1_MODEL = "gemini-3.5-flash"

    def __init__(self, llm_backend=None, api_key=None, model=None, temperature=0.3):
        backend = (os.environ.get("SPARK_LLM_BACKEND", "").strip().lower()
                   or llm_backend or "openai")
        if backend not in DEFAULT_MODELS:
            raise ValueError(f"unknown llm_backend {backend!r}; expected one of {sorted(DEFAULT_MODELS)}")
        self.llm_backend = backend
        self.model = (model or os.environ.get("SPARK_LLM_MODEL")
                      or os.environ.get("SPARK_GEMINI_MODEL") or DEFAULT_MODELS[backend])
        key_var = "GEMINI_API_KEY" if backend == "gemini" else "OPENAI_API_KEY"
        self.api_key = api_key or os.environ.get(key_var, "").strip()
        self.temperature = temperature
        self._client = None
        # dicts of model / prompt / output / thought token counts; the bench reads these.
        self.usage_log = []
        global _backend_logged
        if not _backend_logged:
            _backend_logged = True
            via = os.environ.get("OPENAI_BASE_URL") if backend == "openai" else None
            logger.info("LLM backend %s, model %s%s", backend, self.model, f" via {via}" if via else "")

    def _get_client(self):
        if self._client is not None:
            return self._client
        if self.llm_backend == "gemini":
            if _genai is None:
                raise ImportError("google-genai is not installed (needed for the gemini backend)")
            self._client = _genai.Client(api_key=self.api_key, http_options={"timeout": GEMINI_TIMEOUT_MS})
        else:
            if _OpenAI is None:
                raise ImportError("openai is not installed (needed for the openai backend)")
            self._client = _OpenAI(api_key=self.api_key or "EMPTY", base_url=os.environ.get("OPENAI_BASE_URL"))
        return self._client

    def _generation_config(self, temperature) -> dict:
        """Gemini config. SPARK_GEMINI_THINKING_BUDGET (int) caps thinking tokens; it must be a typed
        ThinkingConfig, the dict form passes validation but does not bind."""
        cfg = {"temperature": temperature}
        budget = os.environ.get("SPARK_GEMINI_THINKING_BUDGET")
        if budget is not None and _genai_types is not None:
            try:
                cfg["thinking_config"] = _genai_types.ThinkingConfig(thinking_budget=int(budget))
            except ValueError:
                logger.warning("SPARK_GEMINI_THINKING_BUDGET=%r is not an int; ignored", budget)
        return cfg

    def _record_usage(self, response, tag: str) -> None:
        """Append the response's token usage to usage_log. Never raises."""
        try:
            u = getattr(response, "usage_metadata", None)
            if u is not None:  # google-genai
                entry = {
                    "tag": tag, "model": self.model,
                    "prompt_tokens": u.prompt_token_count,
                    "output_tokens": u.candidates_token_count,
                    "thought_tokens": getattr(u, "thoughts_token_count", None) or 0,
                    "cached_tokens": getattr(u, "cached_content_token_count", None) or 0,
                }
            else:  # OpenAI-compatible; vLLM reports reasoning tokens in the details
                u = response.usage
                details = getattr(u, "completion_tokens_details", None)
                entry = {
                    "tag": tag, "model": self.model,
                    "prompt_tokens": u.prompt_tokens,
                    "output_tokens": u.completion_tokens,
                    "thought_tokens": getattr(details, "reasoning_tokens", None) or 0,
                    "cached_tokens": 0,
                }
            self.usage_log.append(entry)
            logger.info("[usage] %s %s: in=%s out=%s thoughts=%s", tag, self.model,
                        entry["prompt_tokens"], entry["output_tokens"], entry["thought_tokens"])
        except Exception:  # noqa: BLE001 - usage accounting must not break planning
            pass

    def generate(self, contents, *, tag: str, model=None, temperature=None, json_only=False) -> str:
        """One LLM call; returns the raw reply text.

        `contents` is a str, or a list of str and PIL images / HxWx3 uint8 arrays. `json_only`
        appends a one-line JSON instruction on the openai backend (no response_format: served
        models vary).
        """
        temperature = self.temperature if temperature is None else float(temperature)
        if self.llm_backend == "openai":
            if not model or model.startswith("gemini"):
                model = self.model
            return self._openai_generate(contents, tag, model, temperature, json_only)
        response = self._get_client().models.generate_content(
            model=model or self.model, contents=contents, config=self._generation_config(temperature))
        self._record_usage(response, tag)
        return response.text

    def _openai_generate(self, contents, tag, model, temperature, json_only) -> str:
        parts = []
        for c in [contents] if isinstance(contents, str) else contents:
            if isinstance(c, str):
                parts.append({"type": "text", "text": c})
                continue
            pil = c if isinstance(c, Image.Image) else Image.fromarray(c)
            buf = io.BytesIO()
            pil.convert("RGB").save(buf, format="JPEG", quality=90)
            parts.append({"type": "image_url", "image_url": {
                "url": "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii"),
            }})
        if json_only:
            parts.append({"type": "text", "text": "Output JSON only, no code fences."})
        response = self._get_client().chat.completions.create(
            model=model, temperature=temperature, messages=[{"role": "user", "content": parts}])
        self._record_usage(response, tag)
        return response.choices[0].message.content or ""
