"""通过当前 DINOv2 引擎执行固定 Support 分类。"""
# 固定 Support 可来自确认档案或显式清单，Query 从目录或清单读取。
# 引擎完成注册、预测和可选交换检查后导出结果。
import argparse
import csv
import json
from contextlib import nullcontext
from os.path import commonpath
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from src.gpu_task import gpu_task
from src.inference.engine import create_engine
from src.confirmed_support.engine import READOUT_CONFIRMED
from src.confirmed_support.model_presets import READOUT_CONFIRMED_LORA
from src.confirmed_support.profiles import read_registration_profile
from src.inference.common import sha256_file
from src.task_selection.archives import matching_registration_archives
from src.task_selection.registration import (
    DIFFICULT_PRODUCTS, REVIEW,
    resolve_difficult_registration,
)


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def label_index(value, class_order=('NG', 'OK')) -> int:
    """将实际类别、A/B或0/1映射到当前注册方向。"""
    text = str(value).strip().upper()
    for index, label in enumerate(class_order):
        if text == label.upper():
            return index
    if text in {'A', 'B', '0', '1'}:
        return int(text in {'B', '1'})
    raise ValueError(f'标签不属于当前注册类别 {class_order}：{value}')


def load_queries(*, query_dir=None, query_manifest=None, limit=None, require_labels=False,
                 class_order=('NG', 'OK')):
    """从目录或 CSV/JSON 清单按稳定顺序读取路径、显示标识及可选 GT。"""
    if limit is not None and limit < 1:
        raise ValueError("limit必须为正整数")
    rows = []
    if query_manifest:
        manifest = Path(query_manifest).resolve()
        # CSV 使用 path,label；JSON 使用相同字段的数组或 entries 数组。
        if manifest.suffix.lower() == ".csv":
            with manifest.open(encoding="utf-8-sig", newline="") as stream:
                entries = list(csv.DictReader(stream))
        else:
            raw = json.loads(manifest.read_text(encoding="utf-8-sig"))
            entries = raw["entries"] if isinstance(raw, dict) else raw
        for item in entries:
            path = Path(item["path"])
            path = (manifest.parent / path).resolve() if not path.is_absolute() else path.resolve()
            value = item.get("label")
            label = value if value is not None and str(value).strip() else None
            if class_order and label is not None:
                label = label_index(label, class_order)
            rows.append({"path": str(path), "image_id": str(item.get("image_id", path.name)), "label": label})
    else:
        base = Path(query_dir).resolve()
        if not base.is_dir():
            raise NotADirectoryError(base)
        for path in sorted(base.rglob("*")):
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
                relative = path.relative_to(base)
                # 普通验证目录以 NG/OK 子目录给出 GT；推理目录可以没有标签。
                side = relative.parts[0] if len(relative.parts) > 1 else base.name
                known = class_order and side.upper() in {label.upper() for label in (*class_order, 'A', 'B')}
                label = label_index(side, class_order) if known else side if class_order is None else None
                rows.append({"path": str(path), "image_id": relative.as_posix(), "label": label})
    if limit is not None:
        rows = rows[:limit]
    if not rows:
        raise ValueError("Query输入没有受支持图片")
    paths = [row["path"] for row in rows]
    if len(set(paths)) != len(paths):
        raise ValueError("Query清单包含重复图片路径")
    for row in rows:
        path = Path(row["path"])
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            raise ValueError(f"Query路径不是受支持图片：{path}")
        if require_labels and row["label"] is None:
            raise ValueError(f"验证Query缺少GT标签：{path}")
    return rows


def validation_metrics(rows, class_order=('NG', 'OK')) -> dict:
    """根据逐图 GT 和实际预测统计混淆矩阵、准确率及每类召回率。"""
    confusion = [[0, 0], [0, 0]]
    counts, reviews = [0, 0], [0, 0]
    for row in rows:
        truth, prediction = row['gt_label'], row['prediction']
        counts[truth] += 1
        if prediction is None:
            reviews[truth] += 1
        else:
            confusion[truth][prediction] += 1
    total = sum(counts)
    recalls = [confusion[side][side] / counts[side] if counts[side] else None for side in (0, 1)]
    return {"queries": total, "correct": confusion[0][0] + confusion[1][1],
            "accuracy": (confusion[0][0] + confusion[1][1]) / total,
            "class_order": list(class_order), "confusion_matrix": confusion,
            "class_counts": dict(zip(class_order, counts)),
            "review_count": sum(reviews), "review_by_class": dict(zip(class_order, reviews)),
            "recall": dict(zip(class_order, recalls)),
            "balanced_accuracy": sum(recalls) / 2 if all(value is not None for value in recalls) else None}


def model_options(version,resources,adapter=None):
    """显式选择原默认或独立LoRA研究资源，不自动切换默认。"""
    key=version
    selected=dict(resources[key])
    if version=='custom':
        selected['adapter_checkpoint']=adapter or selected.get('adapter_checkpoint','')
        if not selected['adapter_checkpoint']: raise ValueError('自选研究版必须明确适配器文件')
    selected['readout_mode']=READOUT_CONFIRMED if version=='default' else READOUT_CONFIRMED_LORA
    if version=='default': selected.pop('adapter_checkpoint',None)
    return selected


def support_paths(manifest, class_order=('NG', 'OK')):
    """校验实际固定清单的两类路径及已有字节身份。"""
    manifest = Path(manifest).resolve()
    entries=json.loads(manifest.read_text(encoding='utf-8-sig'))['entries']
    sides = ([], [])
    for item in entries:
        path=(manifest.parent / item['path']).resolve()
        if not path.is_file():
            raise ValueError('固定Support清单包含无效路径或类别')
        if item.get('sha256') and sha256_file(path)!=item['sha256']:
            raise ValueError(f'Support字节身份不符：{path}')
        sides[label_index(item['role'], class_order)].append(str(path))
    if not all(sides) or len(set(sides[0] + sides[1])) != len(entries):
        raise ValueError('Support两类必须非空且图片路径不重复')
    return sides


def _profile_inputs(profile):
    if not profile:
        return []
    return [profile['source_root'], *profile['support_a'], *profile['support_b']]


def _discover_registration(args, resources, route):
    paths = []
    for value in [ROOT / 'runs/registration_sessions', *resources.get('registration_archive_paths', []),
                  *args.archive_path]:
        source = Path(value).resolve()
        paths.extend(sorted(source.rglob('*.json')) if source.is_dir() else [source])
    matches = matching_registration_archives(list(dict.fromkeys(paths)), route.product_label)
    if len(matches) > 1:
        raise ValueError('找到多个本产品注册档案，请通过 --registration 选择：' + '、'.join(map(str, matches)))
    return matches[0] if matches else None


def run(args):
    """选择注册路线，保存逐图结果与实际资源身份。"""
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('推理输出已有内容，拒绝覆盖')
    models_path = Path(args.models)
    resources = json.loads(models_path.read_text(encoding='utf-8-sig')) if models_path.is_file() else {}
    registration = args.registration
    profile = read_registration_profile(registration) if registration else None
    class_order = (args.class_a or 'NG', args.class_b or 'OK')
    if class_order[0].upper() == class_order[1].upper():
        raise ValueError('A和B须指定不同真实类别')
    manual_support = support_paths(args.support_manifest, class_order) if args.support_manifest else None
    queries = load_queries(query_dir=args.query_dir, query_manifest=args.query_manifest, class_order=None)
    route_paths = [args.query_dir, *(query['path'] for query in queries), *_profile_inputs(profile)]
    if manual_support:
        route_paths.extend(manual_support[0] + manual_support[1])
    route = resolve_difficult_registration(
        *route_paths, confirmed_product=args.product,
        allow_existing_generic=bool(registration or manual_support),
    )
    if route.task_type == REVIEW:
        raise ValueError(f'{route.reason}：请核对目录或使用 --product 确认真实产品')
    if not registration and not manual_support and route.product_label:
        registration = _discover_registration(args, resources, route)
        if registration:
            profile = read_registration_profile(registration)
    if profile:
        class_order = (profile['class_a'], profile['class_b'])
        if any(requested and requested != saved for requested, saved in
               zip((args.class_a, args.class_b), class_order)):
            raise ValueError('指定的实际类别与注册档案不一致')
        a, b = profile['support_a'], profile['support_b']
    elif manual_support:
        a, b = manual_support
    else:
        raise ValueError('未找到当前产品的注册档案，请提供 --support-manifest 完成新注册')
    if not all(Path(path).is_file() for path in a + b):
        raise FileNotFoundError('注册档案中的实际Support文件缺失')

    if args.task_type not in ('auto', 'generic'):
        raise ValueError('This release supports generic A/B classification only')
    model_version = args.model_version
    options = model_options(model_version, resources, args.adapter_checkpoint)
    if profile and profile['readout_mode'] != options['readout_mode']:
        raise ValueError('注册档案读出与当前产品路线不一致')

    validating = getattr(args, 'validation', False)
    known_labels = {label.upper() for label in (*class_order, 'A', 'B', '0', '1')}
    if args.query_dir and validating and {label.upper() for label in class_order} != {'NG', 'OK'}:
        queries = [query for query in queries if str(query['label']).upper() in known_labels]
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError('limit必须为正整数')
        queries = queries[:args.limit]
    if not queries:
        raise ValueError('没有符合当前注册类别的Query')
    for query in queries:
        label = query['label']
        if label is not None and str(label).strip() and (args.query_manifest or str(label).upper() in known_labels):
            query['label'] = label_index(label, class_order)
        elif validating:
            raise ValueError('验证Query缺少GT标签：' + query['path'])
        else:
            query['label'] = None
    paths = [row['path'] for row in queries]
    if set(paths) & set(a + b):
        raise ValueError('Query不能与Support路径重合')

    support_mode = 'research'
    scope = nullcontext() if support_mode == 'formal' else gpu_task(args.device)
    profile_to_save = None
    with scope:
        engine = create_engine(device=args.device, support_mode=support_mode, task_type=route.task_type, **options)
        try:
            engine.prepare_task_paths(list(dict.fromkeys(a + b + paths)))
            if profile:
                engine.load_registration(registration)
            else:
                engine.register_exact_support(a, b, verify_swap=True)
            predicted = engine.predict(paths, batch_size=8)
            identity = engine.describe()
            swapped = None
            if args.verify_swap and all(row.predicted_index is not None for row in predicted):
                engine.register_exact_support(b, a, verify_swap=True)
                reverse = engine.predict(paths, batch_size=8)
                delta = max(abs(first.final_logits[index] - second.final_logits[1 - index])
                            for first, second in zip(predicted, reverse) for index in (0, 1))
                agreement = sum(first.predicted_index == 1 - second.predicted_index
                                for first, second in zip(predicted, reverse))
                swapped = {'max_abs_score_column_swap_error': delta, 'physical_prediction_agreement': agreement,
                           'queries': len(paths), 'passed': delta <= 1e-6 and agreement == len(paths)}
                if not swapped['passed']:
                    raise ValueError(f'固定Support实际Query交换检查失败：{swapped}')
        finally:
            engine.clear_registration()

    rows = []
    for query, prediction in zip(queries, predicted):
        a_score, b_score = prediction.final_logits or (None, None)
        row = {'image_id': query['image_id'], 'path': query['path'], 'a_score': a_score, 'b_score': b_score,
               'prediction': prediction.predicted_index, 'predicted_side': prediction.predicted_class}
        if validating:
            row.update(gt_label=query['label'], correct=prediction.predicted_index == query['label'])
        rows.append(row)

    output.mkdir(parents=True, exist_ok=True)
    prefix = ''
    with (output / f'{prefix}predictions.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    weight_version = identity.get('adapter_identity', {}).get('model_version') or 'dinov2_vits14_reg'
    identity.update(model_id=model_version, task_id=route.task_id, weight_version=weight_version)
    identity.update(route_reason=route.reason, product_id=route.product_id, product_label=route.product_label)
    science_status = (
        'PRODUCTION_DEFAULT' if model_version == 'default' else 'SCIENCE_RESEARCH_ONLY'
    )
    result = {'model_version': model_version, 'science_status': science_status, 'identity': identity,
              'class_order': list(class_order), 'registration_path': str(registration or profile_to_save or ''),
              'support_a': a, 'support_b': b, 'queries': len(paths), 'swap_verification': swapped,
              'score_export_precision': 'FP64', 'training_performed': False}
    if validating:
        result['metrics'] = validation_metrics(rows, class_order)
        (output / f'{prefix}metrics.json').write_text(json.dumps(result['metrics'], ensure_ascii=False, indent=2),
                                                    encoding='utf-8')
    result['qp_solver_precision'] = 'FP64'
    (output / f'{prefix}summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    return result


def build_parser(*, validation=False):
    """为推理和普通验证提供同一模型、Support 与 Query 参数合同。"""
    parser=argparse.ArgumentParser(description='固定Support的当前DINOv2模型验证' if validation else __doc__)
    parser.add_argument('--model-version',choices=('default','custom'),default='default',
        help='通用模型默认 default')
    source=parser.add_mutually_exclusive_group()
    source.add_argument('--registration')
    source.add_argument('--support-manifest')
    queries=parser.add_mutually_exclusive_group(required=True)
    queries.add_argument('--query-dir',help='Query目录；验证时使用NG/OK子目录提供GT')
    queries.add_argument('--query-manifest',help='CSV或JSON清单，字段path,label；相对路径以清单目录解析')
    parser.add_argument('--output-dir',required=True)
    parser.add_argument('--models',default=str(ROOT/'configs/models.local.json'))
    parser.add_argument('--adapter-checkpoint',help='显式使用的本机适配权重路径')
    parser.add_argument('--archive-path', action='append', default=[],
                        help='用于发现历史注册的 JSON 文件或目录，可重复指定')
    parser.add_argument('--product', choices=tuple(DIFFICULT_PRODUCTS),
                        help='目录无法自动识别时显式确认真实产品目录')
    parser.add_argument('--task-type', choices=('auto', 'generic'), default='auto')
    parser.add_argument('--class-a', help='新建注册的真实 A 类，默认 NG')
    parser.add_argument('--class-b', help='新建注册的真实 B 类，默认 OK')
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--limit',type=int)
    parser.add_argument('--verify-swap',action='store_true')
    parser.set_defaults(validation=validation)
    return parser


def main(argv=None, *, validation=False):
    """解析输入并执行实际 engine；验证模式同时导出 GT 指标。"""
    result=run(build_parser(validation=validation).parse_args(argv))
    print(json.dumps({'model_version':result['model_version'],'queries':result['queries'],
                      'science_status':result['science_status'],'swap_verification':result['swap_verification'],
                      **({'metrics':result['metrics']} if validation else {})},ensure_ascii=False))
    return 0


if __name__=='__main__': raise SystemExit(main())
