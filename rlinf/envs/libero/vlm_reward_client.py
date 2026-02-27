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

import logging
import os
import tempfile
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import requests

logger = logging.getLogger(__name__)


@dataclass
class LiberoVLMRewardConfig:
    endpoint: str = "http://127.0.0.1:18080/score"
    timeout: float = 120.0
    nframes: int = 16
    max_pixels: int = 256 * 28 * 28
    backend: Optional[str] = None
    backend_kwargs: dict[str, Any] = field(default_factory=dict)
    fps: int = 4
    fail_on_request_error: bool = False
    keep_temp_video: bool = False
    temp_video_dir: Optional[str] = None


class LiberoVLMRewardClient:
    """HTTP client for LIBERO video-based reward scoring."""

    def __init__(self, cfg: LiberoVLMRewardConfig):
        self.cfg = cfg

    @staticmethod
    def _normalize_frame(frame: np.ndarray) -> np.ndarray:
        frame = np.asarray(frame)
        if frame.ndim == 3 and frame.shape[0] == 3 and frame.shape[-1] != 3:
            frame = np.transpose(frame, (1, 2, 0))
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        return frame

    @staticmethod
    def _sample_frames(frames: list[np.ndarray], nframes: int) -> list[np.ndarray]:
        if not frames:
            return []
        if len(frames) <= nframes:
            return frames
        indices = np.linspace(0, len(frames) - 1, nframes, dtype=int)
        return [frames[idx] for idx in indices]

    def _write_video(self, frames: list[np.ndarray]) -> str:
        import imageio.v2 as imageio

        temp_dir = self.cfg.temp_video_dir
        if temp_dir is not None:
            os.makedirs(temp_dir, exist_ok=True)
        fd, video_path = tempfile.mkstemp(suffix=".mp4", dir=temp_dir)
        os.close(fd)

        with imageio.get_writer(video_path, fps=self.cfg.fps) as writer:
            for frame in frames:
                writer.append_data(self._normalize_frame(frame))

        return video_path

    def score_episode(
        self,
        task_text: str,
        frames: list[np.ndarray],
        fallback_score: float = 0.0,
    ) -> float:
        if len(frames) == 0:
            return fallback_score

        sampled_frames = self._sample_frames(frames, self.cfg.nframes)
        video_path = self._write_video(sampled_frames)
        try:
            payload = {
                "task_text": task_text,
                "video_path": video_path,
                "nframes": min(self.cfg.nframes, len(frames)),
                "max_pixels": self.cfg.max_pixels,
            }
            if self.cfg.backend:
                payload["backend"] = self.cfg.backend
            if self.cfg.backend_kwargs:
                payload["backend_kwargs"] = self.cfg.backend_kwargs
            resp = requests.post(
                self.cfg.endpoint,
                json=payload,
                timeout=self.cfg.timeout,
            )
            if resp.status_code != 200:
                err_msg = (
                    f"Reward request failed with status_code={resp.status_code}, "
                    f"body={resp.text[:300]}"
                )
                if self.cfg.fail_on_request_error:
                    raise RuntimeError(err_msg)
                logger.warning("%s; fallback_score=%.4f", err_msg, fallback_score)
                return fallback_score

            try:
                data = resp.json()
            except Exception as exc:
                err_msg = f"Reward response JSON parse failed: {exc}"
                if self.cfg.fail_on_request_error:
                    raise RuntimeError(err_msg) from exc
                logger.warning("%s; fallback_score=%.4f", err_msg, fallback_score)
                return fallback_score

            if "score" not in data:
                err_msg = f"Reward response missing 'score' field: keys={list(data.keys())}"
                if self.cfg.fail_on_request_error:
                    raise RuntimeError(err_msg)
                logger.warning("%s; fallback_score=%.4f", err_msg, fallback_score)
                return fallback_score
            try:
                return float(data["score"])
            except Exception as exc:
                err_msg = f"Reward score cast failed: score={data.get('score')}, err={exc}"
                if self.cfg.fail_on_request_error:
                    raise RuntimeError(err_msg) from exc
                logger.warning("%s; fallback_score=%.4f", err_msg, fallback_score)
                return fallback_score
        except Exception as exc:
            if self.cfg.fail_on_request_error:
                raise
            logger.warning(
                "Reward request exception: %s; fallback_score=%.4f",
                exc,
                fallback_score,
            )
            return fallback_score
        finally:
            if not self.cfg.keep_temp_video and os.path.exists(video_path):
                os.remove(video_path)
