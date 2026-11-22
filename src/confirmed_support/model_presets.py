"""本模块将研究模型名称映射到本机配置的基座、源码和适配器路径。
推理引擎使用返回路径加载资源；用于兼容性检查"""
from pathlib import Path
import json

READOUT_CONFIRMED_LORA = 'dinov2_confirmed_lora_support_margin'
MODEL_KEYS = ('custom',)
LOCAL_CONFIG = Path(__file__).resolve().parents[2] / 'configs/models.local.json'
LORA_CONFIG = {'rank': 4, 'alpha': 4, 'scaling': 1, 'dropout': 0,
               'blocks': [10, 11], 'targets': ['Q', 'V']}


def default_model_resources() -> dict[str, str]:
    """读取本项目安装器写入的默认资源，不回退作者机器路径。"""
    resources = _local_models().get('default', {})
    fields = ('checkpoint', 'official_source')
    if any(not resources.get(field) for field in fields):
        raise ValueError(f'默认模型未配置：{LOCAL_CONFIG}；请配置本机官方资源')
    return {field: str(resources[field]) for field in fields}


def confirmed_model_preset(key: str) -> dict[str, str]:
    """一次选择绑定独立基座、读出和适配器。"""
    if key not in MODEL_KEYS:
        raise ValueError('未知适配器选项：' + key)
    # 每个研究版本同时绑定基座、官方源码和对应适配器。
    resources = _local_models().get(key, {})
    fields = ('checkpoint', 'official_source', 'adapter_checkpoint')
    if any(not resources.get(field) for field in fields):
        raise ValueError(f'研究模型未配置：{key}；请在高级维护设置资源或填写 {LOCAL_CONFIG}')
    return {**{field: str(resources[field]) for field in fields}, 'readout_mode': READOUT_CONFIRMED_LORA}


def _local_models() -> dict:
    """仅显式本机配置提供权重位置，源码不搜索历史研究目录。"""
    return json.loads(LOCAL_CONFIG.read_text(encoding='utf-8')) if LOCAL_CONFIG.is_file() else {}


def configure_custom_model(adapter_checkpoint: str | Path) -> dict[str, str]:
    """显式选择本机适配权重，核验后更新唯一研究映射。"""
    from ..inference.engine import load_confirmed_adapter

    if not str(adapter_checkpoint).strip():
        raise ValueError('请选择本次研究适配权重')
    models = _local_models()
    fields = ('checkpoint', 'official_source')
    resources = next((models.get(key, {}) for key in ('custom', 'default')
                      if all(models.get(key, {}).get(field) for field in fields)), None)
    if resources is None:
        raise ValueError(f'研究模型未配置官方基座和源码；请在高级维护填写 {LOCAL_CONFIG}')
    # 校验适配器后才写入配置，确保自定义条目引用可加载的权重。
    path = Path(adapter_checkpoint).resolve()
    load_confirmed_adapter(path)
    models['custom'] = {**{field: str(resources[field]) for field in fields},
                        'adapter_checkpoint': str(path)}
    LOCAL_CONFIG.write_text(json.dumps(models, ensure_ascii=False, indent=2), encoding='utf-8')
    return confirmed_model_preset('custom')


def adapter_identity(checkpoint: str | Path) -> dict:
    """身份由实际适配资源确定，与待测图片和标签无关。"""
    path = Path(checkpoint).resolve()
    version = 'custom'
    return {'model_version': version, 'checkpoint': str(path),
            'tensor_count': 8, 'parameter_count': 12288}
