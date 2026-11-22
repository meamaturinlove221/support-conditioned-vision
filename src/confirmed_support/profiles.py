"""本模块校验并读写确认注册档案，供后续会话重建 Support 注册。用于兼容性检查。
档案记录模型身份、类别和实际 Support 路径，Query 评价记录独立保存。"""
from __future__ import annotations

import json
from pathlib import Path

from .engine import READOUT_CONFIRMED
from .model_presets import READOUT_CONFIRMED_LORA, LORA_CONFIG, adapter_identity
from .solver import EQUALITY_TOLERANCE, MARGIN_TOLERANCE

FIELDS = ('profile_type', 'format_version', 'readout_mode', 'encoder_id',
          'encoder_checkpoint', 'official_source', 'input_size', 'feature_blocks',
          'precision', 'classification_precision', 'source_root', 'class_a', 'class_b',
          'support_a', 'support_b', 'initial_policy', 'random_seed', 'parent_session',
          'confirmed_session', 'adaptive', 'adaptive_confirmations', 'registration_constraints')


def validate_registration_profile(payload: dict) -> dict:
    """检查本工具实际档案字段；模型载入与约束验收由推理引擎执行。"""
    if not isinstance(payload, dict):
        raise ValueError('注册档案必须是JSON对象')
    if any(key in payload for key in ('queries', 'predictions', 'query_answers', 'prediction_rows')):
        raise ValueError('Query评价记录必须独立保存，不能进入注册档案')
    missing = set(FIELDS) - payload.keys()
    if missing:
        raise ValueError('注册档案缺少字段：' + ', '.join(sorted(missing)))
    state = {field: payload[field] for field in FIELDS}
    # 模式决定档案版本及是否必须绑定适配器，特征形状和精度保持一致。
    mode = state['readout_mode']
    if (state['profile_type'] != 'confirmed_support_registration'
            or mode not in {READOUT_CONFIRMED, READOUT_CONFIRMED_LORA}
            or state['format_version'] != (2 if mode == READOUT_CONFIRMED_LORA else 1)
            or state['encoder_id'] != 'dinov2_vits14_reg' or state['input_size'] != 448
            or state['feature_blocks'] != ['cls_384', 'ordered_patch_1024x384']
            or state['precision'] != 'FP32' or state['classification_precision'] != 'FP64'):
        raise ValueError('注册档案不是官方S/14确认注册模式的实际配置')
    if mode == READOUT_CONFIRMED_LORA:
        adapter = payload.get('adapter_checkpoint')
        if (not adapter or payload.get('adapter_identity') != adapter_identity(adapter)
                or payload.get('lora_config') != LORA_CONFIG):
            raise ValueError('LoRA注册档案缺少实际适配器身份或固定配置')
        state.update(adapter_checkpoint=str(Path(adapter).resolve()),
                     adapter_identity=payload['adapter_identity'], lora_config=payload['lora_config'])
    elif any(payload.get(field) for field in ('adapter_checkpoint', 'adapter_identity', 'lora_config')):
        raise ValueError('原确认注册档案不能附带LoRA适配器')
    if not state['class_a'] or not state['class_b'] or state['class_a'] == state['class_b']:
        raise ValueError('档案必须包含两个不同的真实类别')
    for field in ('support_a', 'support_b'):
        if not isinstance(state[field], list) or not state[field]:
            raise ValueError('档案必须保留A/B全部实际Support路径')
        state[field] = [str(Path(path).resolve()) for path in state[field]]
    # 仅已通过硬间隔求解的注册可写入档案；逐条补图需能对应实际类别 Support。
    constraints = state['registration_constraints']
    if (not constraints.get('success')
            or constraints.get('minimum_support_margin', 0) < 1 - MARGIN_TOLERANCE
            or constraints.get('equality_residual', float('inf')) > EQUALITY_TOLERANCE):
        raise ValueError('只能保存已成功求解的有效确认注册')
    allowed = {state['class_a']: set(state['support_a']), state['class_b']: set(state['support_b'])}
    for item in state['adaptive_confirmations']:
        if str(Path(item['path']).resolve()) not in allowed.get(item['true_class'], set()):
            raise ValueError('补图来源必须属于已确认类别的实际Support')
    return state


def read_registration_profile(path: str | Path) -> dict:
    """只读JSON档案，不加载模型，不读取Query统计或答案。"""
    return validate_registration_profile(json.loads(Path(path).read_text(encoding='utf-8-sig')))


def write_registration_profile(path: str | Path, payload: dict) -> dict:
    """保存成功会话的真实Support和身份，模型释放后亦不必重新加载。"""
    state = validate_registration_profile(payload)
    Path(path).write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding='utf-8')
    return state
