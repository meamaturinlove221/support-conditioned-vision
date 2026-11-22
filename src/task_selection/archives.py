"""按真实产品路径筛选可读注册档案。"""
from pathlib import Path
from src.confirmed_support.profiles import read_registration_profile
from .registration import resolve_difficult_registration


def matching_registration_archives(paths, product_label):
    """仅返回来源及全部注册图片一致绑定该产品的档案。"""
    matches = []
    for path in map(Path, paths):
        try:
            profile = read_registration_profile(path)
            evidence = [profile.get('source_root'), *profile.get('support_a', []), *profile.get('support_b', [])]
            if not evidence[0] or not profile.get('support_a') or not profile.get('support_b'):
                continue
            routes = [resolve_difficult_registration(item) for item in evidence]
            if all(route.product_label == product_label for route in routes):
                matches.append(path)
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return matches
