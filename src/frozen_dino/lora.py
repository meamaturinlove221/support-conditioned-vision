"""在官方 DINO 最后两层 Q/V 投影加入固定 rank 4 的 LoRA。
上游提供官方模型和图像预处理，下游训练或推理复用提取出的双特征块。
适配器单独保存，基座参数和原有读出保持冻结。"""
from __future__ import annotations

import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .features import load_official_model

RANK = 4
ALPHA = 4
BLOCK_INDICES = (10, 11)


class LoRAQKV(nn.Module):
    """包装原qkv线性层，K与基座完全冻结，Q/V各一组低秩残差。"""

    def __init__(self, base: nn.Linear, generator: torch.Generator) -> None:
        """A按固定seed随机初始化、B全零，update0与原qkv等价。"""
        super().__init__()
        if base.in_features != 384 or base.out_features != 1152:
            raise ValueError('LoRA只适配官方ViT-S/14的384维qkv')
        # 固定原投影，只训练 Q/V 的 A、B；B 为零保证初始化时残差为零。
        self.base = base.requires_grad_(False)
        self.q_A = nn.Parameter(torch.empty(RANK, 384))
        self.q_B = nn.Parameter(torch.zeros(384, RANK))
        self.v_A = nn.Parameter(torch.empty(RANK, 384))
        self.v_B = nn.Parameter(torch.zeros(384, RANK))
        nn.init.kaiming_uniform_(self.q_A, a=math.sqrt(5), generator=generator)
        nn.init.kaiming_uniform_(self.v_A, a=math.sqrt(5), generator=generator)

    def forward(self, values: Tensor) -> Tensor:
        """只加Q/V连续残差，保持原K及qkv输出排列。"""
        # 输入末维为 384，原投影按 Q、K、V 各 384 维拼接。
        original = self.base(values)
        # 两次线性变换实现 x A^T B^T，缩放系数为 alpha/rank。
        q = F.linear(F.linear(values, self.q_A), self.q_B) * (ALPHA / RANK)
        v = F.linear(F.linear(values, self.v_A), self.v_B) * (ALPHA / RANK)
        return torch.cat((original[..., :384] + q, original[..., 384:768],
                          original[..., 768:] + v), dim=-1)


def attach_lora(model: nn.Module, seed: int = 42) -> tuple[str, ...]:
    """对严格加载的CPU基座注入唯一两层适配器，仅开放8个新张量。"""
    # 先冻结全部基座，再替换指定注意力层以限定可训练参数。
    model.requires_grad_(False)
    generator = torch.Generator(device='cpu').manual_seed(seed)
    for index in BLOCK_INDICES:
        attention = model.blocks[index].attn
        attention.qkv = LoRAQKV(attention.qkv, generator)
    # 核对实际可训练参数范围，防止误把基座或其他层加入优化器。
    names = tuple(name for name, value in model.named_parameters() if value.requires_grad)
    parameters = dict(model.named_parameters())
    if len(names) != 8 or sum(parameters[name].numel() for name in names) != 12288:
        raise AssertionError('LoRA范围应为8张量／12288参数')
    model.float().eval()
    return names


def build_lora_model(source_path: str, checkpoint_path: str, device: str = 'cpu',
                     seed: int = 42) -> tuple[nn.Module, tuple[str, ...]]:
    """先严格加载官方完整基座，再注入零残差；不使用旧研究权重。"""
    model = load_official_model(source_path, checkpoint_path, 'cpu')
    names = attach_lora(model, seed)
    return model.to(device).float().eval(), names


def adapter_state(model: nn.Module) -> dict[str, Tensor]:
    """仅导出唯一适配器张量，不复制官方基座或分类器参数。"""
    return {name: value.detach().cpu().clone() for name, value in model.named_parameters()
            if name.endswith(('.q_A', '.q_B', '.v_A', '.v_B'))}


def load_adapter(model: nn.Module, state: dict[str, Tensor]) -> None:
    """按明确8个参数键严格恢复适配器，不静默忽略缺失或额外参数。"""
    # 先校验完整键集合与形状，再按对应参数设备和精度复制。
    current = adapter_state(model)
    if set(state) != set(current):
        raise ValueError('LoRA适配器键必须与唯一8张量完全一致')
    parameters = dict(model.named_parameters())
    with torch.no_grad():
        for name, value in state.items():
            if value.shape != current[name].shape:
                raise ValueError(f'LoRA参数尺寸不一致：{name}')
            parameters[name].copy_(value.to(parameters[name]))
