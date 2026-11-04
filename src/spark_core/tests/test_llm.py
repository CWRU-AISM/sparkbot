"""
Offline check of SPARKPlanner.generate (spark_core.llm) on the OpenAI-compatible backend.

No network: the OpenAI client class is replaced by a fake that records the
request. Covers backend and model selection by env, image -> data-URI part, usage
recording under the tag, and the json_only instruction.
"""

import base64
import io
import types

import numpy as np
import pytest
from PIL import Image

from spark_core import llm
from spark_core.planner import SPARKPlanner


class _FakeOpenAI:
    """Stands in for openai.OpenAI. Records kwargs, returns a stub reply."""

    calls = []

    def __init__(self, **kwargs):
        _FakeOpenAI.calls.append({"init": kwargs})
        usage = types.SimpleNamespace(
            prompt_tokens=11, completion_tokens=7,
            completion_tokens_details=types.SimpleNamespace(reasoning_tokens=3),
        )
        reply = types.SimpleNamespace(
            choices=[types.SimpleNamespace(message=types.SimpleNamespace(content='["cup"]'))],
            usage=usage,
        )

        def create(**kw):
            _FakeOpenAI.calls.append({"create": kw})
            return reply

        self.chat = types.SimpleNamespace(
            completions=types.SimpleNamespace(create=create))


def test_generate_openai(monkeypatch):
    monkeypatch.setenv("SPARK_LLM_BACKEND", "openai")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://vllm:8000/v1")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("SPARK_LLM_MODEL", raising=False)  # SPARK_GEMINI_MODEL is only the alias
    monkeypatch.setenv("SPARK_GEMINI_MODEL", "Qwen/Qwen2.5-VL-7B-Instruct")
    monkeypatch.setattr(llm, "_OpenAI", _FakeOpenAI)
    _FakeOpenAI.calls.clear()

    planner = SPARKPlanner()  # default arg: env selects the backend
    assert planner.llm_backend == "openai"

    img = np.zeros((8, 8, 3), dtype=np.uint8)
    out = planner.generate(["find the cup", img], tag="t1", temperature=0,
                           model="gemini-3.5-flash", json_only=True)
    assert out == '["cup"]'

    init = _FakeOpenAI.calls[0]["init"]
    assert init == {"api_key": "EMPTY", "base_url": "http://vllm:8000/v1"}

    req = _FakeOpenAI.calls[1]["create"]
    assert req["model"] == "Qwen/Qwen2.5-VL-7B-Instruct"  # gemini tier collapses
    assert req["temperature"] == 0.0
    parts = req["messages"][0]["content"]
    assert req["messages"][0]["role"] == "user"
    assert parts[0] == {"type": "text", "text": "find the cup"}
    url = parts[1]["image_url"]["url"]
    assert parts[1]["type"] == "image_url" and url.startswith("data:image/jpeg;base64,")
    decoded = Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1])))
    assert decoded.format == "JPEG" and decoded.size == (8, 8)
    assert parts[-1] == {"type": "text", "text": "Output JSON only, no code fences."}

    assert planner.usage_log == [{
        "tag": "t1", "model": "Qwen/Qwen2.5-VL-7B-Instruct",
        "prompt_tokens": 11, "output_tokens": 7, "thought_tokens": 3, "cached_tokens": 0,
    }]

    # The env var wins over the argument (runners select the backend by env).
    assert SPARKPlanner(llm_backend="gemini", api_key="k").llm_backend == "openai"
    monkeypatch.setenv("SPARK_LLM_BACKEND", "gemini")
    assert SPARKPlanner(api_key="k").llm_backend == "gemini"


def test_backend_and_model_defaults(monkeypatch):
    for var in ("SPARK_LLM_BACKEND", "SPARK_LLM_MODEL", "SPARK_GEMINI_MODEL", "OPENAI_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    p = SPARKPlanner()
    assert (p.llm_backend, p.model) == ("openai", llm.DEFAULT_MODELS["openai"])
    assert SPARKPlanner(llm_backend="gemini", api_key="k").model == llm.DEFAULT_MODELS["gemini"]
    monkeypatch.setenv("SPARK_GEMINI_MODEL", "alias-model")
    assert SPARKPlanner().model == "alias-model"
    monkeypatch.setenv("SPARK_LLM_MODEL", "new-model")
    assert SPARKPlanner().model == "new-model"
    assert SPARKPlanner(model="arg-model").model == "arg-model"


def test_unknown_backend_and_family_raise(monkeypatch):
    monkeypatch.setenv("SPARK_LLM_BACKEND", "bogus")
    with pytest.raises(ValueError):
        SPARKPlanner()
    monkeypatch.delenv("SPARK_LLM_BACKEND")
    with pytest.raises(ValueError):
        SPARKPlanner(robot_family="g1")


def test_json_only_off_and_str_contents(monkeypatch):
    monkeypatch.setenv("SPARK_LLM_BACKEND", "openai")
    monkeypatch.setattr(llm, "_OpenAI", _FakeOpenAI)
    _FakeOpenAI.calls.clear()
    SPARKPlanner().generate("hello", tag="t2")
    parts = _FakeOpenAI.calls[1]["create"]["messages"][0]["content"]
    assert parts == [{"type": "text", "text": "hello"}]
