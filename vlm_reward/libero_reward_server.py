# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import argparse
import importlib
import logging
import os
import re
from abc import ABC, abstractmethod
from typing import Any, Callable

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, AutoTokenizer
from vllm import LLM, SamplingParams

logger = logging.getLogger(__name__)


DEFAULT_QWEN_PROMPT_TEMPLATE = """Task info:
{task_instruction}

Question:
Given the task goal and the input video, estimate the current task completion percentage (0-100). The estimation must reflect progress toward the final goal rather than scene description.

Answer format:
Please output a numerical number between 1 and 100 indicating the percentage of task completion.
"""

PRIMO_R1_SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The Assistant is an expert AI specializing in embodied procedure and event reasoning based on visual input (video or images). "
    "The assistant must strictly follow a specific thought process and output format. "
    "The reasoning process is enclosed within <think> </think> tags, and the final answer is within <answer> </answer> tags. "
    "The <think> block must contain three ordered subsections: <planning>, <observation>, and <reasoning>. "
    "The <answer> block must contain only the final output required by the question type and no other commentary."
)

PRIMO_R1_USER_PROMPT = (
    "Task info:\n"
    "{task_instruction}\n\n"
    "Question:\n"
    "Given the task goal and the input video, estimate the current task completion percentage (0-100). The estimation must reflect progress toward the final goal rather than scene description."
    "QUESTION TYPE:\n numerical \n\n"
    "Analyze the provided visual data and reason about the ongoing task.\n\n"
    "Please think about this question as if you were a human pondering deeply. "
    "Provide your detailed reasoning between the <think> and </think> tags, following the subsections <planning>, <observation>, and <reasoning>. "
    "Then give your final answer between the <answer> and </answer> tags.\n\n"
    "Below is the required template:\n\n"
    "<think>\n"
    "<planning>\n"
    "Identify the high-level goal of the agent, what is the initial state? What does successful completion look like?\n"
    "Break down the high-level goal into a logical sequence of canonical steps. This serves as your mental plan for interpreting the task.\n"
    "Use this plan to interpret actions, map observed behaviors to steps, assess progress, detect anomalies, and predict what happens next.\n"
    "</planning>\n"
    "<observation>\n"
    "View the video as a temporal sequence of actions contributing to the procedure.\n"
    "Objectively describe what is occurring in the current moment, noting evidence of progress or state changes.\n"
    "Identify fine-grained actions and explain how they move the task forward.\n"
    "List relevant objects, tools, and environmental context, emphasizing functional states and transformations.\n"
    "Note cues-repetition, transitions, or completion indicators-that situate the action in the procedural script.\n"
    "</observation>\n"
    "<reasoning>\n"
    "Think through the question as a human would, Engage in an internal dialogue using expressions such as 'let me think', 'wait', 'hmm', 'oh, I see', 'let's break it down', etc.\n"
    "Connect observations to the procedural plan to determine which step is being executed, progress, correctness, or anomalies.\n"
    "Reflect on assumptions, verify interpretations, and, if appropriate, predict the agent's next likely action.\n"
    "Synthesize understanding of what the agent is doing, how it fits into the broader task, and whether the process seems successful.\n"
    "You are encouraged to include self-reflection or verification in your reasoning process.\n"
    "</reasoning>\n"
    "</think>\n"
    "<answer>\n"
    "[Final answer here - strictly follow the `Please provide the numerical value (e.g., 42 or 3.14) within the <answer> </answer> tags.` output format and include no extra commentary.]\n"
    "</answer>"
)


def extract_think(output_str: str) -> str:
    pattern = r"<think>\s*(.*?)\s*</think>"
    match = re.search(pattern, output_str, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return ""


def extract_answer(text: str) -> str:
    pattern = r"<answer>\s*(.*?)\s*</answer>"
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return ""


def normalize_number(num_str: str):
    try:
        s = (num_str or "").strip()
        s = s.replace("≈", "").replace("~", "").replace("％", "%")
        if s.endswith("%"):
            s = s[:-1]
            s = s.replace(",", "")
            return float(s) / 100.0
        s = s.replace(",", "")
        return float(s)
    except Exception:
        return None


def _extract_number_from_text(text: str) -> float | None:
    if not text:
        return None

    bracket_match = re.search(r"\[\[\s*([^\]]+?)\s*\]\]", text)
    if bracket_match:
        value = normalize_number(bracket_match.group(1))
        if value is not None:
            return value

    direct_value = normalize_number(text)
    if direct_value is not None:
        return direct_value

    numeric_candidates = re.findall(r"[-+]?\d*\.?\d+\s*%?", text)
    for candidate in reversed(numeric_candidates):
        value = normalize_number(candidate)
        if value is not None:
            return value
    return None


def _parse_score_from_output(output_text: str, score_mode: str = "auto") -> float:
    # Keep this call to make think extraction explicit in logs/debugging flows.
    _ = extract_think(output_text)
    answer_text = extract_answer(output_text)

    # Prefer parsing inside <answer>; this avoids picking numbers from prompts.
    value = _extract_number_from_text(answer_text)
    if value is None and not answer_text:
        # Fallback only when there is no answer tag at all.
        value = _extract_number_from_text(output_text)
    if value is None:
        lower_text = output_text.lower()
        if "success" in lower_text:
            return 1.0
        if "fail" in lower_text:
            return 0.0
        return 0.0

    if score_mode == "percentage":
        score = value / 100.0 if value > 1.0 else value
    elif score_mode == "unit":
        score = value / 100.0 if value > 1.0 else value
    else:  # auto
        score = value / 100.0 if value > 1.0 else value
    return max(0.0, min(1.0, score))


class ScoreRequest(BaseModel):
    task_text: str = Field(min_length=1)
    video_path: str = Field(min_length=1)
    nframes: int = Field(default=16, ge=1, le=128)
    max_pixels: int = Field(default=256 * 28 * 28, ge=28 * 28, le=2048 * 28 * 28)
    backend: str | None = None
    backend_kwargs: dict[str, Any] = Field(default_factory=dict)


class RewardBackend(ABC):
    """Abstract backend interface for task-video reward inference."""

    @abstractmethod
    def score_video(
        self,
        task_text: str,
        video_path: str,
        nframes: int,
        max_pixels: int,
        backend_kwargs: dict[str, Any] | None = None,
    ) -> tuple[float, str]:
        """Return (score, raw_output_text)."""


class QwenVLLMRewardBackend(RewardBackend):
    """Qwen2.5-VL inference backend built on vLLM."""

    def __init__(
        self,
        model_path: str,
        tensor_parallel_size: int,
        max_model_len: int,
        gpu_memory_utilization: float,
        temperature: float,
        top_p: float,
        max_tokens: int,
        prompt_template: str,
    ):
        self.prompt_template = prompt_template
        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            limit_mm_per_prompt={"video": 1, "image": 1},
        )
        self.sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
        self.processor = AutoProcessor.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        tokenizer.padding_side = "left"
        self.processor.tokenizer = tokenizer

    def score_video(
        self,
        task_text: str,
        video_path: str,
        nframes: int,
        max_pixels: int,
        backend_kwargs: dict[str, Any] | None = None,
    ) -> tuple[float, str]:
        backend_kwargs = backend_kwargs or {}
        prompt_template = backend_kwargs.get("prompt_template", self.prompt_template)
        score_mode = backend_kwargs.get("score_mode", "auto")

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": video_path,
                        "max_pixels": max_pixels,
                        "nframes": nframes,
                    },
                    {
                        "type": "text",
                        "text": prompt_template.format(
                            task_instruction=task_text, task=task_text
                        ),
                    },
                ],
            }
        ]
        prompt = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        _, video_inputs, video_kwargs = process_vision_info(
            messages, return_video_kwargs=True
        )
        llm_inputs = [
            {
                "prompt": prompt,
                "multi_modal_data": {"video": video_inputs[0]},
                "mm_processor_kwargs": {k: v[0] for k, v in video_kwargs.items()},
            }
        ]
        outputs = self.llm.generate(llm_inputs, self.sampling_params)
        output_text = outputs[0].outputs[0].text
        return _parse_score_from_output(output_text, score_mode=score_mode), output_text


class PrimoR1VLLMRewardBackend(RewardBackend):
    """PRIMO-R1 backend: first frame + video + last frame."""

    def __init__(
        self,
        model_path: str,
        tensor_parallel_size: int,
        max_model_len: int,
        gpu_memory_utilization: float,
        temperature: float,
        top_p: float,
        max_tokens: int,
        system_prompt: str,
        user_prompt: str,
    ):
        try:
            import cv2  # noqa: F401
            from PIL import Image  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "PRIMO-R1 backend requires `opencv-python` and `Pillow`."
            ) from exc

        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            limit_mm_per_prompt={"image": 2, "video": 1},
        )
        self.sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
        self.processor = AutoProcessor.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        tokenizer.padding_side = "left"
        self.processor.tokenizer = tokenizer

    @staticmethod
    def _extract_first_last_frame(video_path: str):
        import cv2
        from PIL import Image

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video {video_path}")

        try:
            ret, first = cap.read()
            if not ret:
                raise ValueError("Cannot read first frame")

            first = cv2.cvtColor(first, cv2.COLOR_BGR2RGB)
            first_img = Image.fromarray(first)

            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, total - 1))
            ret, last = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, total - 2))
                ret, last = cap.read()

            if not ret:
                last_img = first_img
            else:
                last = cv2.cvtColor(last, cv2.COLOR_BGR2RGB)
                last_img = Image.fromarray(last)
            return first_img, last_img
        finally:
            cap.release()

    def score_video(
        self,
        task_text: str,
        video_path: str,
        nframes: int,
        max_pixels: int,
        backend_kwargs: dict[str, Any] | None = None,
    ) -> tuple[float, str]:
        backend_kwargs = backend_kwargs or {}
        system_prompt = backend_kwargs.get("system_prompt", self.system_prompt)
        user_prompt = backend_kwargs.get("user_prompt", self.user_prompt)
        score_mode = backend_kwargs.get("score_mode", "auto")
        question_text = user_prompt.format(
            task_instruction=task_text,
            task=task_text,
        )
        if "{task_instruction}" not in user_prompt and "{task}" not in user_prompt:
            question_text = f"Task info:\n{task_text}\n\n{question_text}"

        first_img, last_img = self._extract_first_last_frame(video_path)
        messages = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": first_img},
                    {
                        "type": "video",
                        "video": video_path,
                        "max_pixels": max_pixels,
                        "nframes": nframes,
                    },
                    {"type": "image", "image": last_img},
                    {
                        "type": "text",
                        "text": question_text,
                    },
                ],
            },
        ]
        prompt = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages,
            return_video_kwargs=True,
        )
        llm_input = {
            "prompt": prompt,
            "multi_modal_data": {
                "image": image_inputs,
                "video": video_inputs,
            },
        }
        if video_kwargs:
            llm_input["mm_processor_kwargs"] = {
                key: value[0] if isinstance(value, list) and len(value) == 1 else value
                for key, value in video_kwargs.items()
            }

        outputs = self.llm.generate([llm_input], self.sampling_params)
        output_text = outputs[0].outputs[0].text
        return _parse_score_from_output(output_text, score_mode=score_mode), output_text


BackendFactory = Callable[[argparse.Namespace], RewardBackend]
_BACKEND_FACTORIES: dict[str, BackendFactory] = {}


def register_backend(name: str, factory: BackendFactory):
    _BACKEND_FACTORIES[name] = factory


register_backend(
    "qwen2_5_vl_vllm",
    lambda args: QwenVLLMRewardBackend(
        model_path=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        prompt_template=args.prompt_template,
    ),
)
register_backend(
    "PRIMO-R1",
    lambda args: PrimoR1VLLMRewardBackend(
        model_path=args.model_path,
        tensor_parallel_size=args.primo_tensor_parallel_size,
        max_model_len=args.primo_max_model_len,
        gpu_memory_utilization=args.primo_gpu_memory_utilization,
        temperature=args.primo_temperature,
        top_p=args.primo_top_p,
        max_tokens=args.primo_max_tokens,
        system_prompt=args.primo_system_prompt,
        user_prompt=args.primo_user_prompt,
    ),
)
register_backend(
    "primo_r1_vllm",
    lambda args: PrimoR1VLLMRewardBackend(
        model_path=args.model_path,
        tensor_parallel_size=args.primo_tensor_parallel_size,
        max_model_len=args.primo_max_model_len,
        gpu_memory_utilization=args.primo_gpu_memory_utilization,
        temperature=args.primo_temperature,
        top_p=args.primo_top_p,
        max_tokens=args.primo_max_tokens,
        system_prompt=args.primo_system_prompt,
        user_prompt=args.primo_user_prompt,
    ),
)


class LiberoRewardServer:
    def __init__(self, backends: dict[str, RewardBackend], default_backend: str):
        if default_backend not in backends:
            raise ValueError(
                f"default backend '{default_backend}' not found in loaded backends"
            )
        self.backends = backends
        self.default_backend = default_backend

    def score(self, request: ScoreRequest) -> tuple[float, str, str]:
        backend_name = request.backend or self.default_backend
        backend = self.backends.get(backend_name)
        if backend is None:
            raise ValueError(
                f"unknown backend '{backend_name}', available={list(self.backends.keys())}"
            )
        score, raw_output = backend.score_video(
            task_text=request.task_text,
            video_path=request.video_path,
            nframes=request.nframes,
            max_pixels=request.max_pixels,
            backend_kwargs=request.backend_kwargs,
        )
        return score, raw_output, backend_name


def _load_custom_factory(spec: str) -> BackendFactory:
    if ":" not in spec:
        raise ValueError(
            "custom backend factory must be in module_path:symbol_name format"
        )
    module_name, symbol_name = spec.split(":", maxsplit=1)
    module = importlib.import_module(module_name)
    factory = getattr(module, symbol_name)
    if not callable(factory):
        raise TypeError(f"custom backend factory '{spec}' is not callable")
    return factory


def _build_server(args: argparse.Namespace) -> LiberoRewardServer:
    if args.custom_backend_factory:
        register_backend(args.backend, _load_custom_factory(args.custom_backend_factory))

    backend_factory = _BACKEND_FACTORIES.get(args.backend)
    if backend_factory is None:
        raise ValueError(
            f"backend '{args.backend}' is not registered; available={list(_BACKEND_FACTORIES.keys())}"
        )
    backend = backend_factory(args)
    return LiberoRewardServer(backends={args.backend: backend}, default_backend=args.backend)


def build_app(server: LiberoRewardServer):
    app = FastAPI(title="LIBERO VLM Reward Server")

    @app.get("/health")
    async def health():
        return {
            "status": "ok",
            "default_backend": server.default_backend,
            "loaded_backends": list(server.backends.keys()),
        }

    @app.post("/score")
    async def score(req: ScoreRequest):
        if not os.path.exists(req.video_path):
            raise HTTPException(status_code=400, detail="video_path does not exist")
        try:
            score_value, output_text, used_backend = server.score(req)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"inference failed: {exc}") from exc
        return {
            "score": score_value,
            "model_output": output_text,
            "backend": used_backend,
        }

    return app


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Start a LIBERO reward model server. "
            "Default backend is qwen2_5_vl_vllm, and custom backends can be loaded "
            "with --custom-backend-factory."
        )
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="qwen2_5_vl_vllm",
        help="backend name used by /score requests when request.backend is not provided",
    )
    parser.add_argument(
        "--custom-backend-factory",
        type=str,
        default=None,
        help=(
            "Optional custom backend factory in the form module_path:symbol_name. "
            "Factory signature: factory(args: argparse.Namespace) -> backend object "
            "that implements score_video(...)."
        ),
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="Qwen/Qwen2.5-VL-7B-Instruct",
        help="Model path for qwen2_5_vl_vllm backend",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=18080,
    )
    parser.add_argument(
        "--gpu-ids",
        type=str,
        default="0,1",
        help="CUDA_VISIBLE_DEVICES used by this server.",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=2,
        help="for qwen2_5_vl_vllm backend",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=81920,
        help="for qwen2_5_vl_vllm backend",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.8,
        help="for qwen2_5_vl_vllm backend",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="for qwen2_5_vl_vllm backend",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.001,
        help="for qwen2_5_vl_vllm backend",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=64,
        help="for qwen2_5_vl_vllm backend",
    )
    parser.add_argument(
        "--prompt-template",
        type=str,
        default=DEFAULT_QWEN_PROMPT_TEMPLATE,
        help="for qwen2_5_vl_vllm backend",
    )
    parser.add_argument(
        "--primo-tensor-parallel-size",
        type=int,
        default=1,
        help="for PRIMO-R1 backend",
    )
    parser.add_argument(
        "--primo-max-model-len",
        type=int,
        default=16384,
        help="for PRIMO-R1 backend",
    )
    parser.add_argument(
        "--primo-gpu-memory-utilization",
        type=float,
        default=0.9,
        help="for PRIMO-R1 backend",
    )
    parser.add_argument(
        "--primo-temperature",
        type=float,
        default=0.1,
        help="for PRIMO-R1 backend",
    )
    parser.add_argument(
        "--primo-top-p",
        type=float,
        default=0.9,
        help="for PRIMO-R1 backend",
    )
    parser.add_argument(
        "--primo-max-tokens",
        type=int,
        default=1024,
        help="for PRIMO-R1 backend",
    )
    parser.add_argument(
        "--primo-system-prompt",
        type=str,
        default=PRIMO_R1_SYSTEM_PROMPT,
        help="for PRIMO-R1 backend",
    )
    parser.add_argument(
        "--primo-user-prompt",
        type=str,
        default=PRIMO_R1_USER_PROMPT,
        help="for PRIMO-R1 backend",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids

    server = _build_server(args)
    app = build_app(server)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
