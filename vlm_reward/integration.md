# VLM Reward Integration in RLinf (LIBERO-only)

## Scope

This integration is intentionally scoped to **RL with LIBERO Benchmark** in RLinf.
It does not modify the generic RLinf reward pipeline for other tasks.

## What is integrated

- Environment: `LIBERO` (`env.train.task_suite_name=libero_10` for this project).
- VLA: `OpenVLA-OFT`.
- RL algorithm (for this experiment setup): `GRPO`.
- Reward modes supported:
  - Native LIBERO reward (0/1 terminal-style reward)
  - VLM reward with `Qwen2.5-VL-7B-Instruct`
  - VLM reward with `PRIMO-R1`

## Runtime architecture

### 1. Rollout-side frame collection (inside LIBERO env)

File: `rlinf/envs/libero/libero_env.py`

- The env caches step frames per episode (`full_images` from observation).
- On episode end (termination/truncation), it triggers one reward-model scoring call.
- The returned score is converted to step reward through existing LIBERO reward shaping path (`reward_coef`, `use_rel_reward`).

### 2. Client-side episode video packaging

File: `rlinf/envs/libero/vlm_reward_client.py`

- Frames are uniformly sampled up to `nframes=16`.
- If episode has fewer than 16 frames, all frames are used.
- Sampled frames are encoded into a temporary MP4.
- Request payload sent to reward server:
  - `task_text`
  - `video_path`
  - `nframes`
  - `max_pixels` (`256 * 28 * 28 = 200704`)
  - optional `backend` and `backend_kwargs`

### 3. Server-side VLM inference backend

File: `vlm_reward/libero_reward_server.py`

- Provides `POST /score` for video scoring.
- Backend registry supports multiple model-specific inference paths.
- Built-in backends:
  - `qwen2_5_vl_vllm`
  - `PRIMO-R1` (alias: `primo_r1_vllm`)

#### Qwen2.5-VL backend

- Input: video-only message.
- Uses `process_vision_info(...)` and vLLM multimodal input.

#### PRIMO-R1 backend

- Input format: **first frame + video + last frame + text prompt**.
- First/last frames extracted from input video via OpenCV.
- Uses your provided `SYSTEM_PROMPT` / `USER_PROMPT` style parsing with `<think>/<answer>`.

### 4. Output parsing and normalization

File: `vlm_reward/libero_reward_server.py`

- Helper functions:
  - `extract_think(...)`
  - `extract_answer(...)`
  - `normalize_number(...)`
- Score parsing prefers `<answer>...</answer>` content.
- `score_mode` supports:
  - `auto` (default, recommended)
  - `percentage`
  - `unit`

## Config knobs used by LIBERO env

In `examples/embodiment/config/env/libero_*.yaml`:

```yaml
reward_model:
  enable: False
  endpoint: "http://127.0.0.1:18080/score"
  timeout: 120
  nframes: 16
  max_pixels: 200704
  backend: "qwen2_5_vl_vllm"  # or "PRIMO-R1"
  backend_kwargs: {}
  video_fps: 4
  fail_on_request_error: False
  binary_reward: True
  success_threshold: 0.5
  keep_temp_video: False
  temp_video_dir: null
```

## Failure policy (important)

- `fail_on_request_error=False` (default): warning + fallback score.
- `fail_on_request_error=True`: fail fast; useful for strict experiment runs.

## Why this does not conflict with other RLinf features

- Changes are isolated to `LIBERO` env reward path and `vlm_reward` server/client utilities.
- The generic `reward_worker` and rollout data structs were restored to standard logic.
- VLM reward path is opt-in via `reward_model.enable=True`.
