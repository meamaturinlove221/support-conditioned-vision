"""RGB预处理、资产图组索引及既有数据划分。

索引扫描构造不可变的图组和资产记录，划分校验引用其标识。
注册界面通过资产、图组和划分记录组织实际图片路径。
资产目录按完整文件名构建 A/B 配对索引，划分文件按资产或资产族解析，
并检查资产及资产族隔离。RGB 使用既有 BILINEAR 缩放及可选像素缓存。
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
from PIL import Image
import torch
from torch import Tensor


@dataclass(frozen=True, slots=True)
class Group:
    """一个资产图组及按文件名匹配的 A/B 图片路径。"""
    asset_id: str
    family_id: str
    group_id: str
    directory: Path
    pairs: tuple[tuple[Path, Path], ...]

    @property
    def images_a(self) -> tuple[Path, ...]:
        """按配对顺序返回该图组全部 A 类路径。"""
        return tuple(pair[0] for pair in self.pairs)

    @property
    def images_b(self) -> tuple[Path, ...]:
        """按配对顺序返回该图组全部 B 类路径。"""
        return tuple(pair[1] for pair in self.pairs)


@dataclass(frozen=True, slots=True)
class Asset:
    """同一实例所属的资产族与全部图组。"""
    asset_id: str
    family_id: str
    groups: tuple[Group, ...]


@dataclass(frozen=True, slots=True)
class DatasetIndex:
    """数据根目录和按资产标识组织的完整索引。"""
    root: Path
    assets: dict[str, Asset]

    @property
    def groups(self) -> tuple[Group, ...]:
        """按资产与图组的既有顺序展平全部图组。"""
        return tuple(group for asset in self.assets.values() for group in asset.groups)

    @property
    def families(self) -> set[str]:
        """返回索引涉及的资产族集合。"""
        return {asset.family_id for asset in self.assets.values()}


@dataclass(frozen=True, slots=True)
class Split:
    """记录训练、验证和测试资产，以及原划分单位。"""
    train: tuple[str, ...]
    val: tuple[str, ...]
    test: tuple[str, ...]
    unit: str

    def names(self, split: str) -> tuple[str, ...]:
        """返回指定划分的资产标识，拒绝未知划分名称。"""
        try:
            return {"train": self.train, "val": self.val, "test": self.test}[split]
        except KeyError as exc:
            raise ValueError(f"Unknown split: {split}") from exc


_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
_ASSET_NAME = re.compile(r"^.+__\d+$")
_INSTANCE_NAME = re.compile(r"^\d+$")


def image_files(directory: Path) -> tuple[Path, ...]:
    """按路径排序列出目录中support的图片文件。"""
    return tuple(
        sorted(
            path
            for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES
        )
    )


def _read_asset_id(group_dir: Path, root: Path) -> str:
    """依次从目录命名、实例层级或 manifest 解析资产标识。"""
    # 优先使用带实例号的目录名，其次组合上级类别和数字实例目录。
    relative_parts = group_dir.relative_to(root).parts
    for part in reversed(relative_parts[:-1]):
        if _ASSET_NAME.match(part):
            return part
    for index in range(len(relative_parts) - 2, 0, -1):
        if _INSTANCE_NAME.match(relative_parts[index]):
            return f"{relative_parts[index - 1]}__{relative_parts[index]}"

    # 目录未提供明确标识时，依次尝试图组和父目录的元数据。
    for manifest in (group_dir / "manifest.json", group_dir.parent / "manifest.json"):
        if not manifest.is_file():
            continue
        try:
            raw = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for key in ("asset_id", "source_asset", "asset", "object_id"):
            value = raw.get(key)
            if isinstance(value, str) and value.strip():
                return value.replace("/", "__")
    return group_dir.parent.name


def family_from_asset(asset_id: str) -> str:
    """从带实例编号或路径形式的资产标识中提取资产族。"""
    if "__" in asset_id:
        head, tail = asset_id.rsplit("__", 1)
        if tail.isdigit():
            return head
    if "/" in asset_id:
        return asset_id.split("/", 1)[0]
    return asset_id


def _pair_images(a_dir: Path, b_dir: Path) -> tuple[tuple[Path, Path], ...]:
    """按完整文件名匹配 A/B 图像，拒绝空组和文件名缺失。"""
    images_a = image_files(a_dir)
    images_b = image_files(b_dir)
    if not images_a or not images_b:
        raise ValueError(f"Empty A/B image directory: {a_dir.parent}")

    # 比较完整文件名而非仅比较数量，避免错配不同图像。
    by_name_a = {path.name: path for path in images_a}
    by_name_b = {path.name: path for path in images_b}
    names_a, names_b = set(by_name_a), set(by_name_b)
    if names_a != names_b:
        missing_from_a = sorted(names_b - names_a)
        missing_from_b = sorted(names_a - names_b)
        raise ValueError(
            f"A/B filenames differ in {a_dir.parent}; "
            f"missing from A={missing_from_a[:8]}, "
            f"missing from B={missing_from_b[:8]}"
        )
    return tuple(
        (by_name_a[name], by_name_b[name]) for name in sorted(names_a)
    )


def discover_dataset(
    root: str | Path,
    progress: Callable[[int, int, str], None] | None = None,
) -> DatasetIndex:
    """文件扫描和索引扫描 ，root 下的 group/A、group/B，返回索引并通过 progress 报告进度。"""
    root_path = Path(root).resolve()
    if not root_path.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {root_path}")

    if progress is not None:
        progress(0, 0, f"Scanning folders under {root_path}")
    # 先收集候选 A 目录，再只索引有同级 B 的完整图组。
    a_dirs = sorted(path for path in root_path.rglob("A") if path.is_dir())
    total = len(a_dirs)
    if progress is not None:
        progress(0, total, f"Found {total} candidate A directories")

    grouped: dict[str, list[Group]] = {}
    report_every = max(1, total // 200) if total else 1
    for index, a_dir in enumerate(a_dirs, start=1):
        group_dir = a_dir.parent
        b_dir = group_dir / "B"
        if not b_dir.is_dir():
            if progress is not None and (index % report_every == 0 or index == total):
                progress(index, total, f"Indexed {index}/{total} candidate groups")
            continue
        # 图组以资产聚合，组内图像在进入索引前完成严格配对。
        asset_id = _read_asset_id(group_dir, root_path)
        family_id = family_from_asset(asset_id)
        group = Group(
            asset_id=asset_id,
            family_id=family_id,
            group_id=group_dir.relative_to(root_path).as_posix(),
            directory=group_dir,
            pairs=_pair_images(a_dir, b_dir),
        )
        grouped.setdefault(asset_id, []).append(group)
        if progress is not None and (index % report_every == 0 or index == total):
            progress(index, total, f"Indexed {index}/{total} candidate groups")

    if not grouped:
        raise ValueError(f"No group/A + group/B directories found under {root_path}")
    # 对资产和组排序，使下游采样的输入顺序稳定。
    assets = {
        asset_id: Asset(
            asset_id=asset_id,
            family_id=groups[0].family_id,
            groups=tuple(sorted(groups, key=lambda item: item.group_id)),
        )
        for asset_id, groups in sorted(grouped.items())
    }
    if progress is not None:
        progress(total, total, f"Dataset index ready: {len(assets)} assets, {sum(len(a.groups) for a in assets.values())} groups")
    return DatasetIndex(root=root_path, assets=assets)


def load_split(path: str | Path, index: DatasetIndex) -> Split:
    """读取 path 中的既有划分并结合 index 解析，返回经校验的 Split。"""
    split_path = Path(path)
    raw = json.loads(split_path.read_text(encoding="utf-8"))
    unit = str(raw.get("split_unit", "asset"))

    def resolve(name: str) -> tuple[str, ...]:
        """将指定划分的资产列表或资产族展开为已存在的资产。"""
        # 资产族形式先展开到当前索引；两种形式最终都检查引用是否存在。
        assets_key = f"{name}_assets"
        families_key = f"{name}_families"
        if assets_key in raw:
            values = tuple(str(item) for item in raw[assets_key])
        elif families_key in raw:
            families = set(str(item) for item in raw[families_key])
            values = tuple(asset.asset_id for asset in index.assets.values() if asset.family_id in families)
        else:
            raise KeyError(f"split.json lacks {assets_key} or {families_key}")
        missing = sorted(set(values) - set(index.assets))
        if missing:
            preview = missing[:10]
            raise ValueError(f"Split references missing assets: {preview}")
        return tuple(sorted(values))

    split = Split(resolve("train"), resolve("val"), resolve("test"), unit)
    _validate_split(split, index)
    return split


def _validate_split(split: Split, index: DatasetIndex) -> None:
    """检查划分非空、资产完全分配及资产和资产族互不重叠。"""
    # 三个集合必须非空、互斥，并覆盖当前索引的全部资产。
    train, val, test = set(split.train), set(split.val), set(split.test)
    if not train or not val or not test:
        raise ValueError("train/val/test must all contain at least one asset")
    if train & val or train & test or val & test:
        raise ValueError("Asset overlap exists among train/val/test")
    assigned = train | val | test
    unassigned = sorted(set(index.assets) - assigned)
    if unassigned:
        raise ValueError(f"Dataset contains assets absent from split.json: {unassigned[:10]}")
    # 不同实例即使资产名不同，同族跨划分仍会泄漏物体信息。
    family_sets = [{index.assets[name].family_id for name in names} for names in (train, val, test)]
    if family_sets[0] & family_sets[1] or family_sets[0] & family_sets[2] or family_sets[1] & family_sets[2]:
        raise ValueError("Asset-family overlap exists among train/val/test")


class ImagePipeline:
    """按路径及目标尺寸缓存 uint8 RGB，每次返回独立 FP32 张量。"""
    def __init__(self, config) -> None:
        """保存调用方缓存设置，训练在线读取时可关闭缓存。"""
        self.config = config
        self._resized_cache: dict[tuple[Path, int], np.ndarray] = {}

    def load_resized(self, path: str | Path, size: int) -> Tensor:
        """保持现有缩放和数值范围，返回 CHW、0 到 1 的 RGB 张量。"""
        if size < 1:
            raise ValueError("size must be positive")
        source_path = Path(path)
        key = (source_path, int(size))
        array = self._resized_cache.get(key)
        if array is None:
            # RGB 转换与 BILINEAR 缩放沿用既有输入处理，不增加扰动。
            with Image.open(source_path) as source:
                image = source.convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
                array = np.asarray(image, dtype=np.uint8).copy()
            if self.config.cache_images:
                self._resized_cache[key] = array
        return torch.from_numpy(array.copy()).permute(2, 0, 1).float().div_(255.0)

    def stack_resized(self, paths: Iterable[str | Path], size: int) -> Tensor:
        """按输入顺序读取并组成批次，拒绝空输入。"""
        values = [self.load_resized(path, size) for path in paths]
        if not values:
            raise ValueError("No images were selected")
        return torch.stack(values)
