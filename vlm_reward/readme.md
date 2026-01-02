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

按照 rlinf 官网中的步骤配置的环境用于 rlinf 的 quickstart 测试。

**重要提示**：运行 `vlm.py` 部署模型需要使用另一个单独的虚拟环境。这个虚拟环境需要能够运行 qwen2.5-vl 系列模型。以下提供一个精简的环境配置供参考：
```bash
conda create -n xxx python=3.10 -y
conda activate xxx
pip install vllm==0.13.0 ray==2.53.0 transformers==4.57.3 openai>=2.14.0 qwen-vl-utils[decord]==0.0.8 requests protobuf
```

**注**：如果环境配置有问题，可以参考以下资源：
- [Qwen2.5-VL-7B-Instruct 官方文档](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)
- [Video-R1 项目环境配置](https://github.com/tulerfeng/Video-R1)

## 运行代码

**建议**：先将 quickstart 跑通，再进行后续内容。

### 步骤 1: 启动 VLM 服务

`./vlm_reward/vlm.py` 是用于部署 VLM 的代码，其中模型路径、端口都可以修改。

使用前面配置好的虚拟环境，运行 `vlm.py`，默认将 VLM 部署在 `localhost:8000`，并配置模型为 `Qwen/Qwen2.5-VL-7B-Instruct`。具体配置可以在 `vlm.py` 中自行修改。如果进行修改，需要同步修改 `rlinf/workers/reward/reward_worker.py` 中 `init_worker` 函数的相应配置。

**注意**：在运行后续测试代码之前，请先运行 `vlm.py`。

### 步骤 2: 配置奖励模型

对于 `maniskill_ppo_openvlaoft_quickstart.yaml` 配置文件，将最后面的 `use_reward_model` 参数从 `False` 改为 `True`。

然后按照 quickstart 的步骤进行测试。


## 代码修改逻辑

### 1. 数据结构支持 (`./rlinf/data/io_struct.py`)
* **`RolloutResult` 类**: 新增 `video_frames` 字段 (List[List[Any]])，用于在不同 Worker 间传递视频帧。
* **合并与切分**: 修改 `merge_result_list` 和 `_split_single_result_by_group`，增加对 `video_frames` 数据的同步处理逻辑。

### 2. 图像采集 (`./rlinf/workers/rollout/hf/huggingface_worker.py`)
* **`generate` 函数**:
    * 从环境输出 (`env_output["obs"]`) 中提取图像。

### 3. VLM 奖励计算 (`./rlinf/workers/reward/reward_worker.py`)

* **初始化 (`init_worker`)**: 配置本地 API 地址及模型。代码中以 `localhost:8000` 与 `Qwen/Qwen2.5-VL-7B-Instruct` 为例。

* **核心逻辑 (`compute_batch_rewards_with_model`)**:
    * 替换原有计算逻辑，改为基于视觉的 API 打分。
    * **流程**: 输入 RolloutResult → 提取视频/Prompt → 均匀采样 8 帧 → Base64 编码 → 调用 API → 解析 `[SCORE]`。
    * 采样帧数 `num_frames` 默认为 8，可自行修改。

* **辅助函数**: 新增 `_build_api_payload` (构造请求)、`_sample_frames` (采样)、`_image_to_base64` (编码)、`_call_api_and_parse` (正则解析分数)。

## 注意事项

在构造 API 请求时的 prompt 可能需要根据实际任务进行修改，具体请查看 `rlinf/workers/reward/reward_worker.py` 中的 `_build_api_payload` 函数。