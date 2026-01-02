# RLinf-VLM-Reward

本项目基于 rlinf 框架，将 OpenVLA-OFT 算法中的 PPO 奖励机制替换为基于VLM的reward。

## 🛠️ 环境配置

请按照 rlinf 官网指南配置基础环境（推荐使用 `uv` 安装）：

* [安装指南 (Installation)](https://rlinf.readthedocs.io/en/latest/rst_source/start/installation.html)
* [quickstart配置](https://rlinf.readthedocs.io/en/latest/rst_source/start/vla.html)

**提醒：**
在 Quickstart 的 **Step 2** 中，需要修改的配置文件存放位置为：
* `./examples/embodiment/run_embodiment.sh`
* `./examples/embodiment/config/maniskill_ppo_openvlaoft_quickstart.yaml`

按照rlinf官网中的步骤进行配置的环境用于rlinf的quickstart测试，运行vlm.py部署模型需要使用另一个虚拟环境
这个虚拟环境我是直接复用可以运行qwen2.5-vl系列模型的环境，我本人使用的是之前配置好的环境，所以这里我提供一个精简的配置，仅供参考：
```bash
conda create -n xxx python=3.10 -y
conda activate xxx
pip install vllm==0.13.0 ray==2.53.0 transformers==4.57.3 openai>=2.14.0 qwen-vl-utils[decord]==0.0.8 requests protobuf
```

注：如果环境配置有问题，可以参考https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct  中的内容与video-r1这个项目的环境配置：https://github.com/tulerfeng/Video-R1

## 运行代码
建议先将quickstart跑通，再进行后续内容。

./vlm_reward/vlm.py是用于部署vlm的代码，其中模型路径、端口都可以修改。
使用前面自己配置好的xxx环境，运行vlm.py，默认将vlm部署在localhost:8000，并配置模型为Qwen/Qwen2.5-VL-7B-Instruct，具体可以在vlm.py中自行修改。如果进行修改，在后续的代码reward_worker.py中的init_worker函数中也需要修改。

在运行后续测试的代码之前，请先运行vlm.py
（下面部分的内容没再进行测试）
对于maniskill_ppo_openvlaoft_quickstart.yaml最后面的use_reward_model，请把False改为True。
然后按照quickstart的步骤进行测试


## 代码修改逻辑：

### 1. 数据结构支持 (`./rlinf/data/io_struct.py`)
* **`RolloutResult` 类**: 新增 `video_frames` 字段 (List[List[Any]])，用于在不同 Worker 间传递视频帧。
* **合并与切分**: 修改 `merge_result_list` 和 `_split_single_result_by_group`，增加对 `video_frames` 数据的同步处理逻辑。

### 2. 图像采集 (`./rlinf/workers/rollout/hf/huggingface_worker.py`)
* **`generate` 函数**:
    * 从环境输出 (`env_output["obs"]`) 中提取图像。

### 3. VLM 奖励计算 (`./rlinf/workers/reward/reward_worker.py`)
* **初始化 (`init_worker`)**: 配置本地 API 地址及模型.代码中以`localhost:8000`与`Qwen/Qwen2.5-VL-7B-Instruct`为例

* **核心逻辑 (`compute_batch_rewards_with_model`)**:
    * 替换原有计算逻辑，改为基于视觉的 API 打分。
    * **流程**: 输入 RolloutResult -> 提取视频/Prompt -> 均匀采样 8 帧 -> Base64 编码 -> 调用 API -> 解析 `[SCORE]`。
    采样帧数 `num_frames` 默认为 8，可自行修改。

* **辅助函数**: 新增 `_build_api_payload` (构造请求), `_sample_frames` (采样), `_image_to_base64` (编码), `_call_api_and_parse` (正则解析分数).

## 注意事项
在构造请求时的prompt可能需要进行修改，具体请查看`rlinf/workers/reward/reward_worker.py`中的`_build_api_payload`函数。