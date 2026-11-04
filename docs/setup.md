# Setup

## Requirements

- Tested on Ubuntu 22.04 and 24.04 (x86_64) with an NVIDIA driver for CUDA 12.6, git and [uv](https://docs.astral.sh/uv/).
- Headless rendering through EGL: `sudo apt install libegl1 libgl1 libglvnd0`. The NVIDIA driver
  must provide its EGL vendor file (`10_nvidia.json` in glvnd's `egl_vendor.d`).
- Disk: Qwen weights about 16 GB, SAM3 checkpoint about 3.5 GB, LIBERO-PRO assets about 0.6 GB.

## Install

```bash
git submodule update --init   # if you cloned without --recurse-submodules
bash scripts/setup.sh
hf auth login
```

`setup.sh` applies `scripts/sam3.patch` to the SAM3 submodule, runs `uv sync --extra bench --extra dev`
(CUDA torch from the PyTorch index, everything else from `uv.lock`), downloads the LIBERO-PRO
perturbed suites, checks their fingerprint, writes `.libero/config.yaml` and runs an import check.
It never writes the per-user LIBERO config. Re-running it is safe.

The SAM3 checkpoint is gated: accept the license at https://huggingface.co/facebook/sam3 before
`hf auth login`. `smoke.sh` downloads it and says so if access is missing.

## Planner

```bash
bash scripts/serve_planner.sh          # foreground; add & or use a second terminal
```

vLLM runs in its own cached uv environment (it pins its own torch). The first run downloads the
model. Set `VLLM_MEM` (GPU memory fraction) and `MAX_SEQS` for your GPU, `PORT` for another port.
To use a server on another machine, start it there with `bash scripts/serve_planner.sh --host
0.0.0.0` (it binds 127.0.0.1 by default; the last flag wins) or use an SSH tunnel, and export
`OPENAI_BASE_URL=http://<host>:8000/v1` before sourcing `scripts/env.sh`. To use Gemini instead: `uv sync --extra bench --extra gemini`, `export
SPARK_LLM_BACKEND=gemini GEMINI_API_KEY=...` (not the reported protocol).

## Check

```bash
bash scripts/smoke.sh
```

It runs the unit tests, the import and data checks, an EGL context, the SAM3 checkpoint download,
one trial with a scripted plan, one planner call, two LIBERO-PRO trials (`--wrist-verify
--relation-bind` only, not the full Table 1 arm). It passes when every step runs and writes well-formed results; the success counts it prints
are for information only.

## Reproduce

See [reproduce.md](reproduce.md).

## Environment

Every script sources `scripts/env.sh`, which also activates `.venv`. Source it yourself before
running `python -m spark_bench...` by hand. It puts `third_party/LIBERO` on `PYTHONPATH` (that is how the
runs imported `libero`) and sets `PYTHONNOUSERSITE=1`, `LIBERO_CONFIG_PATH=<repo>/.libero`, `MUJOCO_GL=egl`,
`PYOPENGL_PLATFORM=egl` and the planner variables (`SPARK_LLM_BACKEND`, `SPARK_LLM_MODEL`,
`OPENAI_BASE_URL`, `OPENAI_API_KEY`). Any of them can be exported first to override.

## Pins

| Component | Pin |
|---|---|
| LIBERO (the `libero` package imported) | Lifelong-Robot-Learning/LIBERO @ 8f1084e3 |
| LIBERO-PRO (scenes, assets, object classes) | Zxy-MLlab/LIBERO-PRO @ eafdb809 |
| LIBERO-PRO perturbed suites | HF dataset zhouxueyang/LIBERO-Pro @ 3a33dc4c |
| SAM3 code | facebookresearch/sam3 @ f1647e8d + `scripts/sam3.patch` (paper runs: 5eb25fb5) |
| SAM3 checkpoint | facebook/sam3 @ 3c879f39 |
| Gripper close | ends once the fingers settle (paper runs: full 300 steps, `SPARK_FAST_CLOSE=0`) |
| Planner weights | MIRALABS/Qwen3.8-27B-W4A16-AutoRound @ a6754efc |
| Planner server | vllm 0.30.0, transformers 5.17.0 |
| Python packages | `uv.lock` |
| Simulator | python 3.12, torch 2.9.1 (cu126), mujoco 3.4.0, robosuite 1.4.1, numpy 2.3.5 |

`sam3.patch` replaces `pkg_resources`, which setuptools 81 and later removed.
LIBERO-PRO must never be importable as `libero`: the protocol runs the upstream package with the
LIBERO-PRO data, and the runner stops if `libero` resolves into LIBERO-PRO.

## Troubleshooting

- `EGL` or `GLContext` errors: check the apt packages and the NVIDIA EGL vendor file above.
- `No module named 'libero'`, or the run waits for input at import: source `scripts/env.sh`.
- `No module named pkg_resources` from sam3: the patch was not applied; re-run `setup.sh`.
- vLLM fails with a `Stale file handle` error: `serve_planner.sh` keeps compile caches in `$TMPDIR`;
  set `TMPDIR` to a local disk.
- On a cluster node without a Hugging Face token, point `HF_HOME` at a cache that already holds
  the pinned SAM3 and Qwen snapshots and set `HF_HUB_OFFLINE=1`.
- Every planner call fails with 404: `SPARK_LLM_MODEL` differs from the served model name.
- Out of GPU memory in a trial: lower `WORKERS` or `VLLM_MEM`. `run_item.sh` treats such a trial
  as failed and runs it again on the next pass.

## Real world deployment

See [real_world.md](real_world.md).
