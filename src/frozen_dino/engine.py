"""共用冻结 DINOv2 编码、任务特征缓存与已注册分类器的预测调度。"""
from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor

import torch
from torch import Tensor, nn

from ..inference.common import PixelPipeline
from .features import IMAGE_SIZE, MODEL_NAME, extract_features


class TaskFeatures:
    """512MiB GPU特征上限，任务或模型不能共享闭式输入缓存。"""

    def __init__(self):
        """每个引擎独有缓存，默认不保留跨注册轮次的任务特征。"""
        self.paths = frozenset()
        self.values = OrderedDict()
        self.bytes = 0

    def prepare(self, paths):
        """改变任务图像池就释放全部旧特征，重注册相同任务可复用。"""
        selected = frozenset(map(str, paths))
        if selected != self.paths:
            self.clear()
            self.paths = selected

    def get(self, path):
        """只读取当前任务的同模型表示，命中后维持有界LRU顺序。"""
        value = self.values.get(str(path))
        if value is not None:
            self.values.move_to_end(str(path))
        return value

    def put(self, path, blocks):
        """独立复制单图张量，避免切片间接持有整个大批次内存。"""
        path = str(path)
        if path not in self.paths or path in self.values:
            return
        # 克隆单图切片，避免一张缓存图间接保留整批显存。
        value = tuple(block.clone() for block in blocks)
        self.values[path] = value
        self.bytes += sum(block.numel()*block.element_size() for block in value)
        # 超过预算时优先淘汰最近最少使用项，字节数随实际张量同步更新。
        while self.bytes > 512*1024**2:
            _, old = self.values.popitem(last=False)
            self.bytes -= sum(block.numel()*block.element_size() for block in old)

    def clear(self):
        """会话结束或任务切换后释放路径及全部特征。"""
        self.values.clear()
        self.paths = frozenset()
        self.bytes = 0


def _read(pipeline, paths):
    """单线程访问已有像素缓存，避免多线程并发改LRU状态。"""
    return torch.stack([pipeline.load_resized(path, IMAGE_SIZE) for path in paths])


def encoded_batches(model, pipeline, paths, device, batch_size):
    """预取仅领先一个CPU批次，GPU仍串行完整编码所有1024个patch。"""
    batches = [paths[i:i+batch_size] for i in range(0, len(paths), batch_size)]
    # 只提前读取下一批，既重叠 CPU 解码与编码，也限制预取内存。
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix='dino-pixels') as reader:
        future = reader.submit(_read, pipeline, batches[0])
        for index, batch in enumerate(batches):
            rgb = future.result()
            if index+1 < len(batches):
                future = reader.submit(_read, pipeline, batches[index+1])
            yield batch, extract_features(model, rgb.to(device).float())


class FrozenDinoSupportProbe:
    """冻结正式编码器，确认注册子类提供 QP 拟合和分类缓存。"""

    def __init__(self, model: nn.Module, pipeline: PixelPipeline, device: str = 'cpu',
                 *, source_path: str = '', checkpoint_path: str = '') -> None:
        """冻结官方模型并初始化空缓存，分类器由确认注册单独拟合。"""
        self.model = model.to(device).float().eval().requires_grad_(False)
        self.pipeline, self.device = pipeline, torch.device(device)
        self.source_path, self.checkpoint_path = str(source_path), str(checkpoint_path)
        self.feature_width = int(getattr(model, 'embed_dim', 384))
        if self.feature_width != 384:
            raise ValueError('正式Support编码器固定使用DINOv2-S/14的384维表示')
        self.encoder_id = MODEL_NAME
        self.cache = None
        self.encoding_batch_size = 2
        self.task_features = TaskFeatures()

    def prepare_task_paths(self, paths: Sequence[str]) -> None:
        """声明本轮任务图像池，缓存独立于Support拟合与Query标签。"""
        self.task_features.prepare(paths)
        self.cache = None

    @torch.inference_mode()
    def encode(self, paths: Sequence[str]) -> tuple[Tensor, Tensor]:
        """按实测完整批次编码，预取一批像素，仅复用本任务同模型表示。"""
        if not paths:
            raise ValueError('图像列表不能为空')
        # 同任务内复用编码结果；拟合时仍只使用所选 Support。
        found = {str(path): self.task_features.get(path) for path in paths}
        missing = [path for path in paths if found[str(path)] is None]
        if missing:
            for batch, blocks in encoded_batches(self.model, self.pipeline, missing, self.device, self.encoding_batch_size):
                for index, path in enumerate(batch):
                    value = tuple(block[index] for block in blocks)
                    self.task_features.put(path, value)
                    found[str(path)] = value
        # 按原路径顺序堆叠每块特征；重复路径也保留对应输出行。
        return tuple(torch.stack(items) for items in zip(*(found[str(path)] for path in paths)))

    @torch.inference_mode()
    def predict_encoded(self, blocks: tuple[Tensor, Tensor]) -> Tensor:
        """调用当前确认缓存，按原 FP64 精度计算含自由截距的两列分数。"""
        if self.cache is None:
            raise ValueError('请先注册Support')
        return self.cache.logits(blocks)

    @torch.inference_mode()
    def predict(self, paths: Sequence[str]) -> Tensor:
        """按实测批次立即分类释放特征，只返回实际两列Final logits。"""
        if not paths:
            raise ValueError('图像列表不能为空')
        if all(self.task_features.get(path) is not None for path in paths):
            size = self.encoding_batch_size
            return torch.cat([self.predict_encoded(self.encode(paths[i:i+size]))
                              for i in range(0, len(paths), size)])
        output = []
        for batch, blocks in encoded_batches(self.model, self.pipeline, paths, self.device, self.encoding_batch_size):
            for index, path in enumerate(batch):
                self.task_features.put(path, tuple(block[index] for block in blocks))
            output.append(self.predict_encoded(blocks))
        return torch.cat(output)

    def clear(self) -> None:
        """清除当前任务注册状态，不保留其他任务的分类向量。"""
        self.cache = None
        self.task_features.clear()

    def describe(self) -> dict:
        """返回真实模型、预处理、固定读出及当前实际注册数量。"""
        width = self.feature_width
        return {'model_variant': self.readout_mode, 'encoder_id': self.encoder_id,
            'source_path': self.source_path, 'encoder_checkpoint': self.checkpoint_path,
            'checkpoint': self.checkpoint_path, 'readout_mode': self.readout_mode,
            'input_size': IMAGE_SIZE,
            'feature_blocks': [f'cls_{width}', f'ordered_patch_1024x{width}'],
            'feature_dimensions': [width, 1024 * width],
            'encoder_block_count': len(self.model.blocks) if hasattr(self.model, 'blocks') else None,
            'input': 'RGB完整双线性448，ImageNetmean/std', 'precision': 'FP32',
            'model_training': self.model.training, 'network_updates': 0,
            'encoding_batch_size': self.encoding_batch_size,
            'task_feature_cache_bytes': self.task_features.bytes,
            'support_counts': list(self.cache.support_counts) if self.cache else None}
