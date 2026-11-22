"""DINOv2-S/14 默认与 LoRA 引擎的创建、Support 注册、档案和逐图推理。"""
from __future__ import annotations

from pathlib import Path
from collections.abc import Sequence

import torch

from ..confirmed_support.engine import ConfirmedDinoSupportProbe, READOUT_CONFIRMED
from ..confirmed_support.model_presets import READOUT_CONFIRMED_LORA, LORA_CONFIG, adapter_identity
from ..confirmed_support.profiles import read_registration_profile, write_registration_profile
from ..confirmed_support.solver import fit_confirmed_support
from ..frozen_dino.features import load_official_model
from ..frozen_dino.lora import build_lora_model, load_adapter
from ..task_selection.registration import GENERIC
from .common import FORMAL_MODE, RESEARCH_MODE, Pixels, SupportInferenceResults, InferenceRuntime


class ConfirmedDinoInferenceEngine(SupportInferenceResults):
    """官方冻结编码器配合 QP 注册，并保存可重新求解的实际路径。"""

    def __init__(self, checkpoint: str | Path, *, official_source: str | Path,
                 device: str = 'auto', support_mode: str = 'research') -> None:
        """严格加载官方基座，固定矩阵精度后建立尚未注册的确认探针。"""
        if support_mode not in {FORMAL_MODE, RESEARCH_MODE}:
            raise ValueError('support_mode必须为formal或research')
        requested = str(InferenceRuntime._resolve_device(device))
        torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        source, weights = Path(official_source).resolve(), Path(checkpoint).resolve()
        # 完整官方参数严格恢复；只为当前引擎创建独立模型与任务缓存。
        model = load_official_model(source, weights, requested)
        self.probe = ConfirmedDinoSupportProbe(model, Pixels(), requested,
            source_path=str(source), checkpoint_path=str(weights))
        self.checkpoint, self.device = str(weights), self.probe.device
        self.support_mode, self._a, self._b = support_mode, (), ()
        self.registration_profile: dict | None = None

    def register_exact_support(self, a: Sequence[str], b: Sequence[str], *, verify_swap: bool = True) -> None:
        """新的约束解和请求的交换检查有效后才更新实际注册路径。"""
        if self.support_mode == 'formal' and (len(a), len(b)) != (4, 4):
            raise ValueError('正式K4模式要求每类4张Support')
        self.probe.register(a, b, verify_swap=verify_swap)
        self._a = tuple(str(Path(path).resolve()) for path in a)
        self._b = tuple(str(Path(path).resolve()) for path in b)
        self.registration_profile = None

    def save_registration(self, path: str | Path, *, metadata: dict) -> dict:
        """有效注册结合真实类别／会话来源保存，不复制任何Query答案。"""
        if self.probe.cache is None:
            raise ValueError('没有可保存的有效注册')
        state = {**metadata, **self.registration_identity(),
                 'profile_type': 'confirmed_support_registration',
                 'support_a': list(self._a), 'support_b': list(self._b),
                 'registration_constraints': self.probe.cache.diagnostics}
        return write_registration_profile(path, state)

    def registration_identity(self) -> dict:
        """注册档案绑定实际编码资源，子类补充独立适配身份。"""
        identity = self.describe()
        fields = ('readout_mode', 'encoder_id', 'encoder_checkpoint', 'official_source',
                  'input_size', 'feature_blocks', 'precision', 'classification_precision')
        return {**{field: identity[field] for field in fields}, 'format_version': 1}

    @torch.inference_mode()
    def load_registration(self, path: str | Path) -> dict:
        """资源核对、真实重求解和全部Support前向通过后才提交新状态。"""
        state = read_registration_profile(path)
        identity = self.registration_identity()
        if (state['readout_mode'] != identity['readout_mode']
                or state.get('adapter_identity') != identity.get('adapter_identity')
                or state.get('lora_config') != identity.get('lora_config')):
            raise ValueError('保存的注册模型版本或适配器与当前模型不一致')
        if (Path(state['encoder_checkpoint']).resolve() != Path(self.checkpoint).resolve()
                or Path(state['official_source']).resolve() != Path(self.probe.source_path).resolve()):
            raise ValueError('保存的注册资源与当前确认注册模型不一致')
        a, b = tuple(state['support_a']), tuple(state['support_b'])
        if self.support_mode == 'formal' and (len(a), len(b)) != (4, 4):
            raise ValueError('正式K4模式要求每类4张Support')
        for support in a + b:
            if not Path(support).is_file():
                raise FileNotFoundError('档案Support文件不存在：' + support)
        # 档案仅保存路径与身份；重新编码和拟合后用全部 Support 自检。
        blocks = self.probe.encode(a + b)
        labels = torch.tensor([0] * len(a) + [1] * len(b), device=self.device)
        proposed = fit_confirmed_support(blocks, labels)
        logits = proposed.logits(blocks)
        correct = int((logits.argmax(1) == labels).sum())
        if correct != len(labels):
            raise ValueError('档案载入后的真实Support自检未全部通过')
        accepted = {**state, 'profile_path': str(Path(path).resolve()),
                    'support_self_check': {'correct': correct, 'total': len(labels),
                    'minimum_true_margin': proposed.diagnostics['minimum_support_margin']}}
        # 自检通过后同时提交缓存和路径，失败时保留现有注册状态。
        self.probe.cache, self._a, self._b = proposed, a, b
        self.probe.last_registration_failure = None
        self.registration_profile = accepted
        return accepted

    def clear_registration(self) -> None:
        """显式新建注册时清除旧档案来源，不能串用上一任务元数据。"""
        super().clear_registration()
        self.registration_profile = None

    def prepare_task_paths(self, paths: Sequence[str]) -> None:
        """切换实际任务清除注册与档案来源，不改变模型资源。"""
        self.probe.prepare_task_paths(paths)
        self._a = self._b = ()
        self.registration_profile = None

    def describe(self) -> dict:
        """导出真实官方资源、精度、注册约束和当前实际 Support 数量。"""
        identity = self.probe.describe()
        return {**identity, 'engine': type(self).__name__,
            'official_source': self.probe.source_path,
            'runtime_model_class': type(self).__name__, 'final_prediction': identity['readout_mode'],
            'prediction_source': identity['readout_mode'], 'registration_mode': READOUT_CONFIRMED,
            'support_a_count': len(self._a), 'support_b_count': len(self._b),
            'support_mode': self.support_mode, 'training_performed': False,
            'optimizer_loaded': False, 'device': str(self.device),
            'registration_profile': getattr(self, 'registration_profile', None)}


def load_confirmed_adapter(checkpoint: str | Path) -> tuple[dict, dict]:
    """核对实际八张量合同，仅恢复当前适配器及其训练状态元数据。"""
    path = Path(checkpoint).resolve()
    payload = torch.load(path, map_location='cpu', weights_only=True)
    state = payload.get('adapter') if isinstance(payload, dict) else None
    # 第 10、11 层的 Q/V 各有 A、B 两个 rank 4 矩阵，共八张量。
    expected = {f'blocks.{block}.attn.qkv.{target}_{side}': shape
                for block in (10, 11) for target in ('q', 'v')
                for side, shape in (('A', (4, 384)), ('B', (384, 4)))}
    if not isinstance(state, dict) or set(state) != set(expected):
        raise ValueError('Adapter must contain the expected eight named tensors')
    for name, shape in expected.items():
        value = state[name]
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape or not value.is_floating_point():
            raise ValueError(f'Invalid adapter tensor: {name}')
    metadata = payload.get('state', {})
    if not isinstance(metadata, dict):
        raise ValueError('Adapter metadata must be a mapping')
    return state, metadata


class ConfirmedLoRADinoInferenceEngine(ConfirmedDinoInferenceEngine):
    """新建独立模型并冻结八个适配参数，原默认模型不共享状态。"""

    def __init__(self, checkpoint: str | Path, *, official_source: str | Path,
                 adapter_checkpoint: str | Path, device: str = 'auto',
                 support_mode: str = RESEARCH_MODE) -> None:
        """研究图像chunk1；归一、FP64求解和交换验收均复用原注册器。"""
        requested = str(InferenceRuntime._resolve_device(device))
        source, base = Path(official_source).resolve(), Path(checkpoint).resolve()
        state, training_state = load_confirmed_adapter(adapter_checkpoint)
        # 固定 FP32 矩阵精度，避免 TF32 改变确认注册输入特征。
        torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        # 独立构建基座后载入已验证的适配器，再交给确认注册探针。
        model, _ = build_lora_model(str(source), str(base), requested)
        load_adapter(model, state)
        self.probe = ConfirmedDinoSupportProbe(model, Pixels(), requested,
            source_path=str(source), checkpoint_path=str(base))
        self.probe.readout_mode = READOUT_CONFIRMED_LORA
        self.probe.encoding_batch_size = 1
        self.adapter_checkpoint = str(Path(adapter_checkpoint).resolve())
        self.adapter_identity = adapter_identity(adapter_checkpoint)
        self.completed_updates = training_state.get('step')
        if self.completed_updates is None:
            self.completed_updates = None
        self.checkpoint, self.device = str(base), self.probe.device
        self.support_mode, self._a, self._b = support_mode, (), ()
        self.registration_profile = None

    def registration_identity(self) -> dict:
        """档案明确绑定适配器与固定LoRA配置。"""
        return {**super().registration_identity(), 'format_version': 2,
                'adapter_checkpoint': self.adapter_checkpoint,
                'adapter_identity': self.adapter_identity, 'lora_config': LORA_CONFIG}

    def describe(self) -> dict:
        """导出原QP读出与实际研究适配器身份。"""
        return {**super().describe(), 'model_variant': READOUT_CONFIRMED_LORA,
                'registration_mode': READOUT_CONFIRMED_LORA,
                'adapter_checkpoint': self.adapter_checkpoint,
                'adapter_identity': self.adapter_identity,
                'adapter_completed_updates': self.completed_updates, 'lora_config': LORA_CONFIG,
                'research_model': True}


def create_engine(
    checkpoint: str | Path | None = None,
    device: str = "auto",
    *,
    support_mode: str = FORMAL_MODE,
    readout_mode: str = READOUT_CONFIRMED,
    official_source: str | Path | None = None,
    adapter_checkpoint: str | Path | None = None,
    task_type: str | None = None,
):
    """按注册任务构造通用二分类 QP 引擎。"""
    if task_type not in (None, GENERIC) or readout_mode not in (READOUT_CONFIRMED, READOUT_CONFIRMED_LORA):
        raise ValueError('任务与读出不匹配')
    if readout_mode == READOUT_CONFIRMED_LORA:
        return ConfirmedLoRADinoInferenceEngine(checkpoint, official_source=official_source,
            adapter_checkpoint=adapter_checkpoint, device=device, support_mode=support_mode)
    return ConfirmedDinoInferenceEngine(checkpoint, official_source=official_source,
        device=device, support_mode=support_mode)
