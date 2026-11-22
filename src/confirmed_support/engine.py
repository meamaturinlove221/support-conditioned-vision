"""用冻结 DINOv2 表示拟合全部 Support 的硬间隔分类器。"""
from __future__ import annotations

from collections.abc import Sequence
import math

import torch
from torch import Tensor

from ..frozen_dino.engine import FrozenDinoSupportProbe
from .solver import ConfirmedSupportCache, fit_confirmed_support, RegistrationConstraintError

READOUT_CONFIRMED = 'dinov2_confirmed_support_margin'


# A/B 交换沿用原统计、系数和分数数值容差。
SWAP_ATOL = SWAP_RTOL = 1e-6


@torch.inference_mode()
def verify_support_swap(original: ConfirmedSupportCache, blocks: tuple[Tensor, Tensor],
                        labels: Tensor) -> dict:
    """反标签重求同一QP，验证统计、函数反号和Support列互换。"""
    report = {'requested': True, 'executed': True, 'passed': False,
              'scope': 'same_cached_support_and_linear_function',
              'atol': SWAP_ATOL, 'rtol': SWAP_RTOL, 'extra_encoder_calls': 0}
    try:
        # 行序不变，只交换A/B语义，避免把归约顺序改变混入本检查。
        swapped = fit_confirmed_support(blocks, 1 - labels)
    except RegistrationConstraintError as error:
        raise RegistrationConstraintError({**error.diagnostics,
            'message': 'A/B交换求解未满足原注册约束',
            'swap_verification': {**report, 'swapped_registration': error.diagnostics}}) from error
    # 对同一批 Support 比较中心尺度、系数与截距反号以及两列分数互换。
    normal_scores, swapped_scores = original.logits(blocks), swapped.logits(blocks)
    statistics_equal = all(torch.equal(first, second) for first, second in
                           zip((*original.centers, *original.scales),
                               (*swapped.centers, *swapped.scales)))
    coefficient_ok = torch.allclose(original.coefficient, -swapped.coefficient,
                                    atol=SWAP_ATOL, rtol=SWAP_RTOL)
    bias_ok = math.isclose(original.bias, -swapped.bias, abs_tol=SWAP_ATOL, rel_tol=SWAP_RTOL)
    scores_ok = torch.allclose(normal_scores.flip(1), swapped_scores,
                               atol=SWAP_ATOL, rtol=SWAP_RTOL)
    physical_changes = int((swapped_scores.argmax(1) != 1 - normal_scores.argmax(1)).sum())
    report.update(statistics_equal=statistics_equal,
                  coefficient_antisymmetry_max_error=float((original.coefficient + swapped.coefficient).abs().max()),
                  bias_antisymmetry_error=abs(original.bias + swapped.bias),
                  support_score_column_swap_max_error=float((normal_scores.flip(1) - swapped_scores).abs().max()),
                  support_physical_class_changes=physical_changes,
                  support_images=len(labels), swapped_registration=swapped.diagnostics)
    report['passed'] = bool(statistics_equal and coefficient_ok and bias_ok and scores_ok and physical_changes == 0)
    if not report['passed']:
        raise RegistrationConstraintError({'success': False, 'message': 'A/B交换检查未通过',
            'swap_verification': report, 'original_registration': original.diagnostics})
    return report


class ConfirmedDinoSupportProbe(FrozenDinoSupportProbe):
    """先验证新解再提交；失败保留原有效分类函数。"""

    def __init__(self, model, pipeline, device: str = 'cpu', **identity) -> None:
        """固定使用S/14的384维表示，编码资源由完整引擎明确绑定。"""
        super().__init__(model, pipeline, device, **identity)
        if self.feature_width != 384:
            raise ValueError('确认注册模型固定使用DINOv2-S/14')
        self.readout_mode = READOUT_CONFIRMED
        self.last_registration_failure: dict | None = None

    @torch.inference_mode()
    def register_encoded(self, blocks: tuple[Tensor, Tensor], labels: Tensor,
                         *, verify_swap: bool = False) -> None:
        """原约束和请求的交换检查都通过才替换旧有效缓存。"""
        try:
            labels = labels.to(self.device)
            # 新分类器和可选交换检查通过后才替换已有注册状态。
            proposed = fit_confirmed_support(blocks, labels)
            report = (verify_support_swap(proposed, blocks, labels) if verify_swap else
                      {'requested': False, 'executed': False, 'passed': None})
            proposed.diagnostics['swap_verification'] = report
        except RegistrationConstraintError as error:
            error.diagnostics.setdefault('swap_verification',
                {'requested': verify_swap, 'executed': False, 'passed': None})
            self.last_registration_failure = error.diagnostics
            raise
        # 只有求解和所请求的交换检查均通过，才原子替换上一份有效缓存。
        self.cache = proposed
        self.last_registration_failure = None

    @torch.inference_mode()
    def register(self, first: Sequence[str], second: Sequence[str],
                 *, verify_swap: bool = False) -> None:
        """仅编码一次完整Support，交换检查复用这些表示。"""
        if not first or not second:
            raise ValueError('A/B两类都需要注册图')
        labels = torch.tensor([0] * len(first) + [1] * len(second), device=self.device)
        self.register_encoded(self.encode(tuple(first) + tuple(second)), labels,
                              verify_swap=verify_swap)

    def describe(self) -> dict:
        """分别记录ViT精度、小规模求解精度与实际约束检查。"""
        identity = super().describe()
        return {**identity, 'classifier': 'hard_margin_linear_with_free_intercept',
            'solver_precision': 'FP64', 'classification_precision': 'FP64',
            'registration_constraints': self.cache.diagnostics if self.cache else None,
            'last_registration_failure': self.last_registration_failure}
