"""本模块从本地官方源码和权重构建 DINOv2，并提取完整图像表示。
输入为 RGB 448 张量，下游读出分别接收 CLS 和按空间顺序排列的 patch 特征。"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import Tensor, nn

MODEL_NAME = 'dinov2_vits14_reg'
IMAGE_SIZE = 448
FEATURE_SHAPES = ((384,), (1024, 384))


def load_official_model(source_path: str | Path, checkpoint_path: str | Path,
                        device: str = 'cpu') -> nn.Module:
    """只从指定官方本地源码构造模型，严格加载完整原始预训练参数。"""
    source = Path(source_path).resolve()
    weights = Path(checkpoint_path).resolve()
    if not (source / 'dinov2/hub/backbones.py').is_file():
        raise FileNotFoundError(source / 'dinov2/hub/backbones.py')
    sys.path.insert(0, str(source))
    from dinov2.hub.backbones import dinov2_vits14_reg
    # 先在 CPU 严格恢复完整权重，再迁移到目标设备并冻结参数。
    model = dinov2_vits14_reg(pretrained=False)
    state = torch.load(weights, map_location='cpu', weights_only=True)
    model.load_state_dict(state, strict=True)
    return model.to(device).float().eval().requires_grad_(False)


def extract_features_train(model: nn.Module, rgb: Tensor) -> tuple[Tensor, Tensor]:
    """共享原预处理与特征定义，eval状态下保留合法LoRA梯度。"""
    if model.training:
        raise ValueError('DINOv2共享特征口径必须保持eval')
    if rgb.ndim != 4 or tuple(rgb.shape[1:]) != (3, IMAGE_SIZE, IMAGE_SIZE):
        raise ValueError('DINOv2输入必须为完整RGB448张量')
    mean = rgb.new_tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
    std = rgb.new_tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
    # 完整 RGB 448 经 ImageNet 标准化，保留 CLS 和全部有序 patch。
    output = model.forward_features((rgb.float() - mean) / std)
    blocks = (output['x_norm_clstoken'], output['x_norm_patchtokens'])
    # 448 / 14 = 32，故 patch 网格为 32×32；register token 不进入返回块。
    width = int(getattr(model, 'embed_dim', blocks[0].shape[-1]))
    if width != 384 or [tuple(block.shape[1:]) for block in blocks] != list(FEATURE_SHAPES):
        raise ValueError('官方S/14 register模型须保留384维CLS及1024个384维有序patch')
    if any(block.dtype != torch.float32 or not torch.isfinite(block).all() for block in blocks):
        raise FloatingPointError('DINOv2特征必须为有限FP32')
    return blocks


@torch.inference_mode()
def extract_features(model: nn.Module, rgb: Tensor) -> tuple[Tensor, Tensor]:
    """只读推理入口保持 inference_mode，图像编码不建立梯度图。"""
    return extract_features_train(model, rgb)
