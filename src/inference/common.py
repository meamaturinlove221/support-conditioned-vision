"""推理与手动会话的公共类型、像素读取、设备选择和结果封装。"""
from __future__ import annotations

from dataclasses import dataclass
from collections import OrderedDict
from collections.abc import Sequence
import hashlib
from pathlib import Path
from time import perf_counter
from typing import Protocol

import numpy as np
from PIL import Image
import torch
from torch import Tensor

FORMAL_MODE = "formal"
RESEARCH_MODE = "research"


@dataclass(frozen=True, slots=True)
class InferencePrediction:
    """保存 A/B 类别、概率、耗时及当前注册分类器的实际两列分数。"""

    query: str
    predicted_class: str
    predicted_index: int | None
    confidence: float
    probability_a: float
    probability_b: float
    milliseconds: float
    support_contexts: int = 1
    swap_consistent: bool | None = None
    final_logits: tuple[float, float] | None = (0.0, 0.0)
    final_prediction: str = ""
    task_type: str = 'generic'
    predicted_count: int | None = None
    normal_count: int | None = None
    count_scores: tuple[float, float, float] | None = None
    review_reason: str | None = None


def sha256_file(path: str | Path) -> str:
    """读取文件摘要，仅供既有 Support 清单和训练权重身份字段核对。"""
    # 分块读取沿用原合同；不生成额外身份文件，也不用于普通图片缓存。
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class PixelPipeline(Protocol):
    """编码器只依赖按指定尺寸读取 RGB FP32 张量的接口。"""

    def load_resized(self, path: str, size: int) -> Tensor:
        """返回三通道、0～1 范围的完整双线性缩放图像。"""
        ...


class Pixels:
    """共用正式 RGB 双线性读取协议，最近像素张量最多保留 64 MiB。"""

    def __init__(self) -> None:
        """像素缓存仅按实际路径和尺寸区分，不保存网络特征。"""
        self._cache: OrderedDict[tuple[str, int], Tensor] = OrderedDict()
        self._bytes = 0

    def load_resized(self, path: str, size: int) -> Tensor:
        """完整 RGB 图像双线性缩放后，按原顺序转为 0～1 FP32 张量。"""
        key = (str(Path(path).resolve()), size)
        if key not in self._cache:
            source_path = Path(path)
            if not source_path.is_file():
                raise FileNotFoundError(source_path)
            # 先用 uint8 复制解码结果，再执行与训练一致的 CHW、FP32 和除 255。
            with Image.open(source_path) as source:
                image = source.convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
                array = np.asarray(image, dtype=np.uint8).copy()
            value = torch.from_numpy(array).permute(2, 0, 1).float().div_(255.0)
            self._cache[key] = value
            self._bytes += value.numel() * value.element_size()
            # 单线程预取访问本缓存；预算满时按最近使用顺序释放 CPU 像素。
            while self._bytes > 64 * 1024**2:
                _, discarded = self._cache.popitem(last=False)
                self._bytes -= discarded.numel() * discarded.element_size()
        self._cache.move_to_end(key)
        return self._cache[key]


class SupportInferenceResults:
    """将当前 QP 探针的两列分数封装为 UI 和 CLI 共用结果。"""

    def predict(self, paths: Sequence[str], *, batch_size: int = 8) -> list[InferencePrediction]:
        """保留 A/B 判定、softmax、逐图耗时及正式 Support/Query 分离规则。"""
        if not self._a or not self._b:
            raise RuntimeError("请先注册Support")
        if batch_size < 1:
            raise ValueError("batch_size必须为正数")
        if self.support_mode == FORMAL_MODE and {str(Path(p).resolve()) for p in paths} & set(self._a + self._b):
            raise ValueError("当前正式Query路径不得与Support重合")
        started = perf_counter()
        # 保留原QP的FP64分数；严格大于才选择B，平分归A。
        logits = self.probe.predict(paths).cpu()
        probabilities = torch.softmax(logits, dim=-1)
        elapsed = (perf_counter() - started) * 1000 / max(1, len(paths))
        rows = []
        for path, values, probs in zip(paths, logits.tolist(), probabilities.tolist()):
            index = int(values[1] > values[0])
            side = "A" if index == 0 else "B"
            rows.append(InferencePrediction(
                query=str(path), predicted_class=side, predicted_index=index,
                confidence=float(probs[index]), probability_a=float(probs[0]),
                probability_b=float(probs[1]), milliseconds=elapsed,
                final_logits=(float(values[0]), float(values[1])), final_prediction=side,
            ))
        return rows

    def clear_registration(self) -> None:
        """释放当前任务的分类缓存和路径，下个任务必须重新注册。"""
        self.probe.clear()
        self._a = self._b = ()


@dataclass(frozen=True, slots=True)
class Candidate:
    """保存候选图像路径、所属图组和 A/B 类别。"""
    path: str
    group_id: str
    side: str


@dataclass(frozen=True, slots=True)
class Task:
    """描述结构化资产或普通类别文件夹构成的手动评估任务。"""
    task_id: str
    candidates_a: tuple[Candidate, ...]
    candidates_b: tuple[Candidate, ...]
    source_mode: str
    group_aware: bool
    source_root: str
    class_a: str = ""
    class_b: str = ""

    def candidates(self, side: str) -> tuple[Candidate, ...]:
        """返回指定 A/B 类别的候选图像。"""
        if side == "A":
            return self.candidates_a
        if side == "B":
            return self.candidates_b
        raise ValueError("side must be A or B")


@dataclass(frozen=True, slots=True)
class Prediction:
    """单图预测、注册数量及实际运行资源。"""
    model_name: str
    checkpoint: str
    query: str
    group_id: str
    true_side: str
    predicted_side: str
    correct: bool | None
    confidence: float
    probability_a: float
    probability_b: float
    support_count_a: int
    support_count_b: int
    milliseconds: float
    task_id: str = ''
    class_a: str = ''
    class_b: str = ''
    source_root: str = ''
    readout_mode: str = ''
    registration_policy: str = ''
    precision: str = ''
    adapter_checkpoint: str = ''
    final_logits_a: float = 0
    final_logits_b: float = 0


@dataclass(frozen=True, slots=True)
class Metrics:
    """记录单个模型的准确率、错误数和两类召回率。"""
    model_name: str
    checkpoint: str
    queries: int
    correct: int | None
    errors: int | None
    accuracy: float | None
    recall_a: float | None
    recall_b: float | None


@dataclass(frozen=True, slots=True)
class RunResult:
    """汇总一次手动注册会话的任务、Support、指标与预测。"""
    task: Task
    policy: str
    support_a: tuple[str, ...]
    support_b: tuple[str, ...]
    metrics: tuple[Metrics, ...]
    predictions: tuple[Prediction, ...]
    session_root: str
    identities: tuple[dict, ...] = ()


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


class InferenceRuntime:
    """为正式引擎提供相同的运行设备和图片入口。"""

    device: torch.device

    @staticmethod
    def _resolve_device(requested: str) -> torch.device:
        """解析 auto/cpu/cuda 并检查显式请求的 CUDA 是否可用。"""
        name = requested.strip().lower()
        if name == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if name == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false")
            return torch.device("cuda")
        if name == "cpu":
            return torch.device("cpu")
        raise ValueError("device must be auto, cuda, or cpu")

    @staticmethod
    def is_image(path: str | Path) -> bool:
        """按扩展名判断文件是否属于support的图片类型。"""
        return Path(path).suffix.lower() in IMAGE_SUFFIXES

    @classmethod
    def image_files(cls, directory: str | Path, *, recursive: bool = False) -> list[Path]:
        """按路径排序枚举目录图片，可选择递归遍历。"""
        root = Path(directory)
        if not root.is_dir():
            raise NotADirectoryError(root)
        iterator = root.rglob("*") if recursive else root.iterdir()
        return sorted(path for path in iterator if path.is_file() and cls.is_image(path))
