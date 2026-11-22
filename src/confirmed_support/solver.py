"""本模块将人工确认的两类 Support 特征拟合为硬间隔线性读出。
为support分类算法的主要算法文件，本质是专门为DINOv2提取的两块特征+二分类标签设计的注册间隔分类器。
上游dinov2编码器提供两块逐图特征与 0/1 标签，下游 Query 复用注册统计和分类向量。
特征统计采用 FP32，小规模对偶求解与分类归约采用 FP64。"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.optimize import minimize
import torch
from torch import Tensor

EQUALITY_TOLERANCE = 1e-7
MARGIN_TOLERANCE = 1e-6


def _polish_active(hessian: np.ndarray, y: np.ndarray, alpha: np.ndarray) -> tuple[np.ndarray, float, dict]:
    """精解原QP的活动集；负乘子退出活动等式，所有原约束仍保留。"""
    # alpha 大于阈值的样本暂时视为“活动点”：它们在理想解中通常恰好落在间隔边界上。
    active = alpha > 1e-8
    # 由当前解先估一个截距，供后续数值精修失败时回退，并记录精修前的状态。
    projection = (hessian @ alpha) * y
    old_bias = float(np.mean(y[active] - projection[active])) if active.any() else 0.0
    report = {'method': 'fixed_active_KKT', 'applied': False,
              'original_minimum_support_margin': float((y * (projection + old_bias)).min()),
              'original_equality_residual': float(abs(y @ alpha)),
              'original_support_margins': (y * (projection + old_bias)).tolist(),
              'original_bias': old_bias, 'original_alpha': alpha.tolist()}
    if not active.any():
        return alpha, old_bias, report
    removed = []
    # 活动集线性系统来自 KKT 条件：活动样本的间隔等式成立，且 y·alpha=0。
    # 如果求出的活动乘子为负，违反 alpha>=0，因此移除该点后重新解；原始样本约束并未删除。
    for _ in range(int(active.sum())):
        if not active.any():
            return alpha, old_bias, {**report, 'reason': 'empty active set'}
        ya = y[active]
        system = np.block([[hessian[np.ix_(active, active)], ya[:, None]],
                           [ya[None, :], np.zeros((1, 1))]])
        solution = np.linalg.lstsq(system, np.r_[np.ones(active.sum()), 0.0], rcond=None)[0]
        candidate = np.zeros_like(alpha)
        candidate[active], bias = solution[:-1], float(solution[-1])
        negative = np.flatnonzero(candidate < 0)
        if not len(negative):
            break
        index = int(negative[np.argmin(candidate[negative])])
        removed.append({'index': index, 'negative_multiplier': float(candidate[index])})
        active[index] = False
    if not active.any():
        return alpha, old_bias, {**report, 'reason': 'empty active set'}
    # KKT 残差：活动点要求恰在 margin=1；非活动点只要求 margin>=1。
    residual = hessian @ candidate + y * bias - 1
    valid = (np.isfinite(solution).all() and candidate.min() >= 0
             and abs(y @ candidate) <= EQUALITY_TOLERANCE
             and np.max(np.abs(residual[active])) <= MARGIN_TOLERANCE
             and np.all(residual[~active] >= -MARGIN_TOLERANCE))
    report.update(applied=bool(valid), kkt_active_residual=float(np.max(np.abs(residual[active]))),
                  kkt_inactive_minimum=float(residual[~active].min()) if (~active).any() else None,
                  removed_active_multipliers=removed)
    return (candidate, bias, report) if valid else (alpha, old_bias, report)


class RegistrationConstraintError(ValueError):
    """注册失败携带真实求解残差，调用者保留上一有效状态。"""

    def __init__(self, diagnostics: dict) -> None:
        """保留失败证据而不假装已经证明数据不可分。"""
        super().__init__('本次注册约束未满足：' + diagnostics['message'])
        self.diagnostics = diagnostics


@dataclass(frozen=True)
class ConfirmedSupportCache:
    """当前任务的Support统计、同一线性分类函数和数值证明。"""

    # 每个特征块各自保存注册时的类别平衡中心和整体尺度；Query 必须复用这些统计量。
    centers: tuple[Tensor, ...]
    scales: tuple[Tensor, ...]
    # 拼接归一化特征后的线性分类向量，以及不受范数惩罚的自由截距。
    coefficient: Tensor
    bias: float
    # (A 类数量, B 类数量)，用于记录本次实际注册规模。
    support_counts: tuple[int, int]
    # 优化器状态和逐张 Support 的约束检查结果，便于报告注册成功或失败的依据。
    diagnostics: dict

    def logits(self, blocks: tuple[Tensor, ...]) -> Tensor:
        """Query仅读取注册统计，FP32特征经FP64线性归约得到两列。"""
        # 展平每个 Query 特征块，使用注册阶段算出的中心/尺度标准化，再沿特征维拼接。
        values = torch.cat([(block.flatten(1).float() - center) / scale
                            for block, center, scale in zip(blocks, self.centers, self.scales)], 1)
        # 有符号分数 m>0 偏向标签0/A，m<0 偏向标签1/B；用 FP64 计算分类分数。
        margin = values.double() @ self.coefficient + self.bias
        # 两列分数的差等于 margin；argmax 因而对应 margin 的正负方向。
        return torch.stack((margin / 2, -margin / 2), -1)

    @property
    def bytes(self) -> int:
        """只统计当前注册长期保留的张量，不计任何Query缓存。"""
        # 统计持久化分类状态占用，不把临时特征或输入图片算进来。
        return sum(v.numel() * v.element_size() for v in (*self.centers, *self.scales, self.coefficient))


@torch.no_grad()
def fit_confirmed_support(blocks: tuple[Tensor, ...], labels: Tensor) -> ConfirmedSupportCache:
    """输入两块批次特征及一维 0/1 标签，输出包含归一化统计和线性系数的缓存。
两块归一后解无 C 的硬间隔对偶，所有 Support 满足约束才返回。"""
    # 标签统一为整数编码；后续约定 0→y=+1（A）、1→y=-1（B）。
    labels = labels.long()
    # 此注册器只接受两个特征块和一维二分类标签，避免输入结构不明确时继续求解。
    if len(blocks) != 2 or labels.ndim != 1 or not torch.all((labels == 0) | (labels == 1)):
        raise ValueError('确认注册需要两块特征及一维A/B标签')
    # 记录两类 Support 数量。每类至少一张，才能定义两侧分类约束和类别平衡权重。
    counts = tuple(int((labels == side).sum()) for side in (0, 1))
    if min(counts) == 0:
        raise ValueError('A/B两类都必须具有Support')
    # 每类总权重均为 1/2；数量不平衡时，少数类的单张图片权重更大。
    weights = torch.where(labels == 0, 1 / (2 * counts[0]), 1 / (2 * counts[1])).float()
    centers, scales, normalized = [], [], []
    # 两个特征块分别求中心和尺度，避免不同块的数值量级直接决定分类贡献。
    for block in blocks:
        # 保留批次维，把每张图片剩余维度展平为一行特征。
        flat = block.flatten(1).float()
        # 确认每行对应一个标签；同设备便于后续张量运算；拒绝 NaN/Inf。
        if len(flat) != len(labels) or flat.device != labels.device or not torch.isfinite(flat).all():
            raise ValueError('Support数量、设备或有限特征不符合合同')
        # 类别平衡加权均值作为此特征块的中心。
        center = (weights[:, None] * flat).sum(0)
        centered = flat - center
        # 按整块所有维度的加权平方变化求一个公共尺度，而非逐维标准化。
        # clamp_min 防止零方差块除零；乘 sqrt(2) 与两块特征的整体归一约定相配。
        scale = (weights[:, None] * centered.square()).sum().sqrt().clamp_min(1e-6) * math.sqrt(2)
        centers.append(center)
        scales.append(scale)
        # 用该块的中心和尺度标准化，留待与其他特征块拼接。
        normalized.append(centered / scale)
    # 合并所有标准化特征块；每行现在是一个 Support 的完整分类特征向量。
    features = torch.cat(normalized, 1)
    # 高精度仅用于当前Support与分类向量；不缓存全部数据集的double特征。
    precise = features.double()
    # Gram 矩阵保存 Support 两两内积：G[i,j]=z_i·z_j，是对偶问题所需的核矩阵。
    gram = (precise @ precise.T).cpu().numpy()
    # 将标签换成 SVM 常用的 +1/-1 符号编码。
    y = (1 - 2 * labels).double().cpu().numpy()
    # H[i,j]=y_i*y_j*(z_i·z_j)，构成对偶目标的二次项矩阵。
    hessian = gram * y[:, None] * y[None, :]
    # 求解硬间隔线性分类器的对偶问题：
    # min 1/2 alpha^T H alpha - sum(alpha)，约束 alpha>=0 且 y^T alpha=0。
    # 不含软间隔惩罚参数 C；SLSQP 的成功标志后面仍需与实际 margin 自检共同验收。
    result = minimize(lambda a: .5 * a @ hessian @ a - a.sum(), np.zeros(len(y)),
        jac=lambda a: hessian @ a - np.ones(len(y)), method='SLSQP',
        bounds=[(0, None)] * len(y),
        constraints={'type': 'eq', 'fun': lambda a: y @ a, 'jac': lambda a: y},
        options={'ftol': 1e-10, 'maxiter': 1000})
    alpha = result.x
    if result.success:
        # 优化器成功后，尝试基于活动 Support 解 KKT 方程，消除有限精度导致的微小间隔误差。
        alpha, bias, polish = _polish_active(hessian, y, alpha)
    else:
        # 优化器失败时只估算诊断所需的截距，不允许后续把失败结果当作有效注册。
        active = alpha > 1e-8
        projection = gram @ (alpha * y)
        bias = float(np.mean(y[active] - projection[active])) if active.any() else 0.0
        polish = {'method': 'fixed_active_KKT', 'applied': False, 'reason': 'SLSQP failed'}
    # 重新识别最终解中的活动点，并从 Support 的加权组合还原线性分类向量：
    # coefficient = sum_i alpha_i*y_i*z_i。
    active = alpha > 1e-8
    coefficient = precise.T @ torch.as_tensor(alpha * y, device=features.device)
    # 自检使用部署时相同的系数点积，而不是仅以QP Gram推导结果代替。
    # 这样检查的是最终返回分类器实际采用的运算形式，而不仅是优化器内部的等价表达。
    margins = y * (precise @ coefficient + bias).cpu().numpy()
    # 汇总可解释的求解证据：收敛状态、对偶等式误差、每张 Support 的 margin、精度和精修信息。
    diagnostics = {'solver': 'SciPy SLSQP', 'success': bool(result.success),
        'message': str(result.message), 'iterations': int(result.nit),
        'objective': float(.5 * alpha @ hessian @ alpha - alpha.sum()),
        'slsqp_objective': float(result.fun),
        'equality_residual': float(abs(y @ alpha)), 'minimum_alpha': float(alpha.min()),
        'minimum_support_margin': float(margins.min()), 'support_margins': margins.tolist(),
        'support_vector_count': int(active.sum()), 'equality_tolerance': EQUALITY_TOLERANCE,
        'margin_tolerance': MARGIN_TOLERANCE, 'ftol': 1e-10, 'maxiter': 1000,
        'feature_statistics_precision': 'FP32', 'solver_precision': 'FP64',
        'classification_precision': 'FP64', 'bias': bias, 'numerical_polish': polish}
    # 只有优化器报告成功、数值有限、对偶等式与非负乘子满足容差，
    # 且每一张 Support 的有符号 margin 都不低于 1-tolerance，才接受注册。
    valid = (result.success and np.isfinite(alpha).all() and np.isfinite(margins).all()
             and diagnostics['equality_residual'] <= EQUALITY_TOLERANCE
             and alpha.min() >= -1e-8 and margins.min() >= 1 - MARGIN_TOLERANCE)
    if not valid:
        # 失败归因证据产物提交
        raise RegistrationConstraintError(diagnostics)
    # 返回 Query 推理所需的归一化统计、分类向量、截距以及注册验收记录。
    return ConfirmedSupportCache(tuple(centers), tuple(scales), coefficient, bias, counts, diagnostics)
