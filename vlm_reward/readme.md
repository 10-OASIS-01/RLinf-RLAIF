# RLinf LIBERO VLM Reward

这个目录现在是 **LIBERO 专用** 的视频 reward 方案。当前内置了 `qwen2_5_vl_vllm` 和 `PRIMO-R1` 两个 backend，服务端是 **可插拔** 的，可以继续接入其它 reward model inference 代码。

- 集成架构说明：`vlm_reward/integration.md`

- 输入：LIBERO rollout 的逐帧视频
- 输出：任务完成分数（默认二值化为 0/1）
- 仅在 `env.reward_model.enable=True` 时启用，默认关闭，不影响其他 RLinf 功能

## 1. 启动 Reward 服务（2 卡）

服务脚本：`vlm_reward/libero_reward_server.py`

- 默认 backend：`qwen2_5_vl_vllm`
- 内置 backend：`qwen2_5_vl_vllm`、`PRIMO-R1`（别名 `primo_r1_vllm`）
- 内置 Qwen 推理：`vllm.LLM(..., tensor_parallel_size=2)` + `qwen_vl_utils.process_vision_info(...)`
- 支持通过 `--custom-backend-factory` 注入自定义 backend（不同模型可用不同 inference 实现）

PRIMO-R1 额外依赖：

- `opencv-python`
- `Pillow`

示例命令：

```bash
python vlm_reward/libero_reward_server.py \
  --backend qwen2_5_vl_vllm \
  --model-path Qwen/Qwen2.5-VL-7B-Instruct \
  --gpu-ids 0,1 \
  --tensor-parallel-size 2 \
  --port 18080
```

默认接口：

- `POST /score`
- body: `{"task_text": "...", "video_path": "...", "nframes": 16, "max_pixels": 200704, "backend": "qwen2_5_vl_vllm", "backend_kwargs": {...}}`

`backend_kwargs` 里可选：

- `score_mode`: `auto`(默认) / `percentage` / `unit`

PRIMO-R1 启动示例（首帧+视频+尾帧）：

```bash
python vlm_reward/libero_reward_server.py \
  --backend PRIMO-R1 \
  --model-path <PRIMO-R1-MODEL-PATH> \
  --gpu-ids 0,1 \
  --primo-tensor-parallel-size 1 \
  --port 18080
```

自定义 backend 示例（将 `my_pkg.my_backend:build_backend` 作为工厂）：

```bash
python vlm_reward/libero_reward_server.py \
  --backend my_custom_backend \
  --custom-backend-factory my_pkg.my_backend:build_backend \
  --port 18080
```

工厂签名约定：

- `build_backend(args: argparse.Namespace) -> backend`
- `backend` 需要实现：
  - `score_video(task_text, video_path, nframes, max_pixels, backend_kwargs) -> (score, raw_output_text)`

## 2. 在 LIBERO 配置中启用

已在 LIBERO 环境配置里增加字段（默认关闭）：

```yaml
reward_model:
  enable: True
  endpoint: "http://127.0.0.1:18080/score"
  timeout: 120
  nframes: 16
  max_pixels: 200704 # 256 * 28 * 28
  backend: "qwen2_5_vl_vllm"  # 或 "PRIMO-R1"
  backend_kwargs: {}
  video_fps: 4
  fail_on_request_error: False
  binary_reward: True
  success_threshold: 0.5
```

对应文件例如：

- `examples/embodiment/config/env/libero_10.yaml`
- `examples/embodiment/config/env/libero_goal.yaml`

## 3. 当前 reward 流程（LIBERO）

在 `LiberoEnv` 内：

1. 每 step 缓存当前帧（用于拼成 episode 视频）
2. 当 episode 结束（termination/truncation）时：
   - 按 `nframes=16` 均匀采样（若总帧数不足 16，则用全部帧）
   - 先写成 mp4 再调用 reward 服务打分
3. 分数映射为 step reward（再按 `reward_coef` 和 `use_rel_reward` 走原有 RL 逻辑）

请求失败策略：

- `fail_on_request_error=False`：记录 warning 并回退到 fallback score（默认）
- `fail_on_request_error=True`：直接抛错中断，避免静默失败

代码入口：

- `rlinf/envs/libero/libero_env.py`
- `rlinf/envs/libero/vlm_reward_client.py`
