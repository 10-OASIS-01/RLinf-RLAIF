# LIBERO-10 Reward Model Experiments (OpenVLA-OFT + GRPO)

- Reward model integration docs: [vlm_reward/readme.md](./vlm_reward/readme.md)
- RLinf original README: [Rlinf_readme.md](./Rlinf_readme.md)

This README is the **task playbook** for reproducing our coauthor experiments.

## 1. Experiment Goal

Run RL training on **LIBERO Long / `libero_10`** and compare reward settings:

1. Native LIBERO reward (0/1).
2. `Qwen2.5-VL-7B-Instruct` as reward model.
3. `PRIMO-R1` as reward model.

Target evidence: in training reward curves, show a trend where **PRIMO-R1 > 0/1 reward** (not necessarily fully converged).

## 2. Fixed Experimental Setup

- Task benchmark: `env.train.task_suite_name=libero_10` (Long).
- VLA model: `OpenVLA-OFT`.
- RL algorithm: `GRPO`.
- Use default settings from RLinf LIBERO docs unless explicitly overridden:
  - [RL with LIBERO Benchmark](https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/libero.html)

## 3. Prerequisites

### 3.1 Checkpoints (provided by us)

Coauthor needs to place these checkpoints on their server:

- OpenVLA-OFT base / SFT checkpoint (for training init)
- Qwen2.5-VL-7B-Instruct checkpoint
- PRIMO-R1 checkpoint

Use your local absolute paths in commands below.

### 3.2 Environment

Follow RLinf official LIBERO setup doc first:

- [RL with LIBERO Benchmark](https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/libero.html)

Additional packages for PRIMO-R1 reward backend:

```bash
pip install opencv-python Pillow
```

## 4. Key Files

- Reward server: `vlm_reward/libero_reward_server.py`
- Reward client: `rlinf/envs/libero/vlm_reward_client.py`
- LIBERO env reward integration: `rlinf/envs/libero/libero_env.py`
- LIBERO env config template: `examples/embodiment/config/env/libero_10.yaml`

## 5. Base Config (directly use existing GRPO config)

Use this existing config directly:

- `examples/embodiment/config/libero_10_grpo_openvlaoft.yaml`

Before training, only edit checkpoint paths in that file:

- `actor.model.model_path`
- `rollout.model.model_path`
- `actor.tokenizer.tokenizer_model`

## 6. Three Runs

All runs use the same GRPO config and differ only in env reward settings.

### Run A: Native LIBERO 0/1 reward (baseline)

In config:

```yaml
env:
  train:
    reward_model:
      enable: False
  eval:
    reward_model:
      enable: False
```

Train:

```bash
bash examples/embodiment/run_embodiment.sh libero_10_grpo_openvlaoft
```

### Run B: Qwen2.5-VL reward

Start reward server:

```bash
python vlm_reward/libero_reward_server.py \
  --backend qwen2_5_vl_vllm \
  --model-path <QWEN2.5-VL-7B-INSTRUCT-PATH> \
  --gpu-ids 0,1 \
  --tensor-parallel-size 2 \
  --port 18080
```

Set reward in config (train/eval both if needed):

```yaml
reward_model:
  enable: True
  endpoint: "http://127.0.0.1:18080/score"
  nframes: 16
  max_pixels: 200704
  backend: "qwen2_5_vl_vllm"
  backend_kwargs:
    score_mode: auto
  fail_on_request_error: True
  binary_reward: True
  success_threshold: 0.5
```

Train:

```bash
bash examples/embodiment/run_embodiment.sh libero_10_grpo_openvlaoft
```

### Run C: PRIMO-R1 reward

Start reward server:

```bash
python vlm_reward/libero_reward_server.py \
  --backend PRIMO-R1 \
  --model-path <PRIMO-R1-PATH> \
  --gpu-ids 0,1 \
  --primo-tensor-parallel-size 1 \
  --port 18080
```

Set reward in config:

```yaml
reward_model:
  enable: True
  endpoint: "http://127.0.0.1:18080/score"
  nframes: 16
  max_pixels: 200704
  backend: "PRIMO-R1"
  backend_kwargs:
    score_mode: auto
  fail_on_request_error: True
  binary_reward: True
  success_threshold: 0.5
```

Train:

```bash
bash examples/embodiment/run_embodiment.sh libero_10_grpo_openvlaoft
```

## 7. Logging and Curves

Each run writes logs under:

- `logs/<timestamp>/`
- TensorBoard events in that run directory.

Compare training reward curves across the three runs.
Suggested reported metric:

- episode return / reward trend over training steps.

Use TensorBoard for quick comparison:

```bash
tensorboard --logdir logs
```

## 8. Minimal Deliverable for Paper Figure

You do not need full convergence. Stop at a moderate budget once trend is clear.

Expected qualitative result:

- `PRIMO-R1 reward` shows better upward trend than `0/1 reward` baseline on `libero_10`.

## 9. Troubleshooting

- If reward server errors should fail the run, keep:
  - `fail_on_request_error: True`
- If reward model output format is unstable, keep:
  - `backend_kwargs.score_mode: auto`
- If PRIMO run fails with import error, install:
  - `opencv-python`, `Pillow`

## 10. Repro Checklist (for coauthor)

1. Install RLinf + LIBERO dependencies from official doc.
2. Place 3 checkpoints (OpenVLA-OFT, Qwen2.5-VL, PRIMO-R1).
3. Edit existing config `libero_10_grpo_openvlaoft.yaml` with your model paths.
4. Run A/B/C with only reward setting changed.
5. Export and compare reward curves.
