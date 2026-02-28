# LIBERO-10 Reward Model Experiment Manual (Single Doc)

This is the only document your collaborator needs to run the full experiment pipeline.

- RLinf original project README: [Rlinf_readme](./Rlinf_readme)
- Reward integration architecture detail: [vlm_reward/integration.md](./vlm_reward/integration.md)

## 1. Goal

Run RL training on `libero_10` (`OpenVLA-OFT + GRPO`) and compare 3 reward settings:

1. Native LIBERO reward (0/1 baseline)
2. Qwen2.5-VL reward
3. PRIMO-R1 reward

Target evidence: reward curve trend where PRIMO-R1 is better than 0/1 baseline (full convergence is not required).

## 2. Fixed Setup

- Benchmark: `env.train.task_suite_name=libero_10`
- Policy: `OpenVLA-OFT`
- RL algorithm: `GRPO`
- Training launcher: `bash examples/embodiment/run_embodiment.sh <config_name>`

## 3. How the Reward Pipeline Works

For VLM reward runs, scoring is done at episode end in LIBERO env:

1. Env caches frame sequence (`full_images`) each step.
2. On termination/truncation, env calls reward client with sampled episode video.
3. Client writes temporary MP4 and requests `POST /score`.
4. Reward server runs selected backend (`qwen2_5_vl_vllm` or `PRIMO-R1`).
5. Score returns to env and is mapped into RL reward using existing `reward_coef` and `use_rel_reward` logic.

Main implementation files:

- Env integration: `rlinf/envs/libero/libero_env.py`
- Reward client: `rlinf/envs/libero/vlm_reward_client.py`
- Reward server: `vlm_reward/libero_reward_server.py`

## 4. Dependencies

First complete RLinf + LIBERO base environment setup from official docs.

- RL with LIBERO Benchmark: [RL with LIBERO Benchmark](https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/libero.html)

Then install VLM reward server dependencies.

Required server dependencies:

- `vllm`
- `transformers`
- `fastapi`
- `uvicorn`
- `qwen-vl-utils[decord]`

Recommended install (run at repo root):

```bash
bash vlm_reward/install_deps.sh
```

The script runs:

```bash
uv sync --extra sglang-vllm --active
```

PRIMO-R1 additional dependencies (only PRIMO backend needs these):

- `opencv-python`
- `Pillow`

Install with:

```bash
bash vlm_reward/install_deps.sh --with-primo
```

## 5. Checkpoints You Must Prepare

Use local absolute paths for all model checkpoints:

- OpenVLA-OFT base/SFT checkpoint (for actor + rollout init)
- Qwen2.5-VL-7B-Instruct checkpoint
- PRIMO-R1 checkpoint

## 6. Config Files Used in This Manual

Base config:

- `examples/embodiment/config/libero_10_grpo_openvlaoft.yaml`

Pre-created experiment configs (already in repo):

- `examples/embodiment/config/libero_10_grpo_openvlaoft_runA.yaml`
- `examples/embodiment/config/libero_10_grpo_openvlaoft_runB_qwen.yaml`
- `examples/embodiment/config/libero_10_grpo_openvlaoft_runC_primo.yaml`

Before any run, set model paths in each run config:

- `actor.model.model_path`
- `rollout.model.model_path`
- `actor.tokenizer.tokenizer_model`

## 7. Reward Server API and Startup

Default endpoint:

- `POST http://127.0.0.1:18080/score`

Request body fields:

- `task_text`
- `video_path`
- `nframes`
- `max_pixels`
- optional `backend`
- optional `backend_kwargs`

Health check:

```bash
curl http://127.0.0.1:18080/health
```

Qwen server:

```bash
python vlm_reward/libero_reward_server.py \
  --backend qwen2_5_vl_vllm \
  --model-path <QWEN2.5-VL-7B-INSTRUCT-PATH> \
  --gpu-ids 0,1 \
  --tensor-parallel-size 2 \
  --max-model-len 16384 \
  --port 18080
```

PRIMO server:

```bash
python vlm_reward/libero_reward_server.py \
  --backend PRIMO-R1 \
  --model-path <PRIMO-R1-PATH> \
  --gpu-ids 0,1 \
  --primo-tensor-parallel-size 1 \
  --port 18080
```

## 8. Run the 3 Experiments

Use separate terminals when a reward server is required.

### 8.1 Run A: Native LIBERO reward (baseline)

Config:

- `examples/embodiment/config/libero_10_grpo_openvlaoft_runA.yaml`
- `env.train.reward_model.enable=False`
- `env.eval.reward_model.enable=False`

Start:

```bash
bash examples/embodiment/run_embodiment.sh libero_10_grpo_openvlaoft_runA
```

### 8.2 Run B: Qwen2.5-VL reward

Config:

- `examples/embodiment/config/libero_10_grpo_openvlaoft_runB_qwen.yaml`
- backend is fixed as `qwen2_5_vl_vllm`

Step 1 (Terminal-1): start Qwen reward server (see Section 7).  
Step 2 (optional): run health check.  
Step 3 (Terminal-2): start training:

```bash
bash examples/embodiment/run_embodiment.sh libero_10_grpo_openvlaoft_runB_qwen
```

### 8.3 Run C: PRIMO-R1 reward

Config:

- `examples/embodiment/config/libero_10_grpo_openvlaoft_runC_primo.yaml`
- backend is fixed as `PRIMO-R1`

Step 1: stop Qwen server if still running.  
Step 2 (Terminal-1): start PRIMO reward server (see Section 7).  
Step 3 (optional): run health check.  
Step 4 (Terminal-2): start training:

```bash
bash examples/embodiment/run_embodiment.sh libero_10_grpo_openvlaoft_runC_primo
```

## 9. Logs and Curve Comparison

Each run writes under configured logger path (default `../results`).

Suggested comparison metric:

- episode return / reward trend over training steps

TensorBoard:

```bash
tensorboard --logdir ../results
```

## 10. Troubleshooting

1. Reward server import errors
- Run `bash vlm_reward/install_deps.sh`
- For PRIMO also run `bash vlm_reward/install_deps.sh --with-primo`

2. Reward service unavailable during training
- Check server process is alive
- Verify `curl http://127.0.0.1:18080/health` works
- Verify config `endpoint` matches running port

3. Strict failure policy
- Keep `fail_on_request_error: True` in runB/runC configs if you prefer fail-fast behavior

4. Output parsing instability
- Keep `backend_kwargs.score_mode: auto`

## 11. Repro Checklist

1. Complete RLinf + LIBERO base setup.
2. Install VLM reward dependencies (`vlm_reward/install_deps.sh`).
3. Prepare 3 checkpoints (OpenVLA-OFT, Qwen2.5-VL, PRIMO-R1).
4. Fill model paths in runA/runB/runC config files.
5. Run A/B/C in order and keep logs separate.
6. Compare curves and report trend.
