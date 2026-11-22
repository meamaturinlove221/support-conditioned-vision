"""手动注册、评估与结果导出。"""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable
import csv
import json
import random

from src.data import Asset, DatasetIndex, discover_dataset, load_split
from .common import Candidate, Task, Prediction, Metrics, RunResult, IMAGE_SUFFIXES
from .engine import create_engine


def index_release(
    dataset_root: str | Path,
    split_path: str | Path | None = None,
    *,
    progress=None,
) -> tuple[DatasetIndex, dict[str, tuple[str, ...]]]:
    """索引发布目录，并在提供有效划分时返回各划分的资产列表。"""
    index = discover_dataset(dataset_root, progress=progress)
    # 有有效划分文件时保留 train/val/test，未提供时仅开放全资产集合。
    split_file = Path(split_path) if split_path and str(split_path).strip() else None
    if split_file and split_file.is_file():
        split = load_split(split_file, index)
        groups = {
            "train": split.train,
            "val": split.val,
            "test": split.test,
            "all": tuple(index.assets),
        }
    else:
        groups = {"train": (), "val": (), "test": (), "all": tuple(index.assets)}
    return index, groups


def task_from_asset(asset: Asset, source_root: str | Path) -> Task:
    """将资产全部图组的 A/B 图片转换为保留图组信息的任务。"""
    rows_a: list[Candidate] = []
    rows_b: list[Candidate] = []
    # 每张图携带原图组标识，逐组留出时可避免同组信息进入注册。
    for group in asset.groups:
        rows_a.extend(Candidate(str(p.resolve()), group.group_id, "A") for p in group.images_a)
        rows_b.extend(Candidate(str(p.resolve()), group.group_id, "B") for p in group.images_b)
    return Task(
        task_id=asset.asset_id,
        candidates_a=tuple(rows_a),
        candidates_b=tuple(rows_b),
        source_mode="structured_release",
        group_aware=True,
        source_root=str(Path(source_root).resolve()),
    )


def discover_class_folders(root: str | Path) -> tuple[str, ...]:
    """列出至少包含一张图片的类别子目录，并要求至少两类。"""
    base = Path(root).resolve()
    if not base.is_dir():
        raise NotADirectoryError(base)
    names = [
        child.name
        for child in sorted(base.iterdir())
        if child.is_dir()
        and any(p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES for p in child.rglob('*'))
    ]
    if len(names) < 2:
        raise RuntimeError(f"分类根目录至少需要2个含图片的子文件夹：{base}")
    return tuple(names)


def task_from_class_folders(root: str | Path, class_a: str, class_b: str) -> Task:
    """将两个不同类别文件夹的全部图片组成 A/B 任务。"""
    if not class_a or not class_b or class_a == class_b:
        raise ValueError("A类和B类必须选择两个不同文件夹")
    base = Path(root).resolve()
    # 普通目录没有真实图组关系，分别按所选类别建立候选列表。
    rows_a = _folder_candidates(base / class_a, "A")
    rows_b = _folder_candidates(base / class_b, "B")
    return Task(
        task_id=f"{class_a}__{class_b}",
        candidates_a=tuple(rows_a),
        candidates_b=tuple(rows_b),
        source_mode="class_folders",
        group_aware=False,
        source_root=str(base),
        class_a=class_a,
        class_b=class_b,
    )


def validate_support(task: Task, selected: Iterable[str | Path], side: str) -> list[str]:
    """校验所选路径属于当前类别，保持顺序去重并拒绝空选择。"""
    # 路径先解析为绝对路径，再验证所属类别并按原选择顺序去重。
    allowed = {row.path for row in task.candidates(side)}
    out: list[str] = []
    seen: set[str] = set()
    for value in selected:
        path = str(Path(value).resolve())
        if path not in allowed:
            raise ValueError(f"{side}注册图不属于当前任务：{path}")
        if path not in seen:
            out.append(path)
            seen.add(path)
    if not out:
        raise ValueError(f"{side}类至少选择1张注册图")
    return out


def path_to_group(task: Task) -> dict[str, str]:
    """建立任务全部候选图片到图组标识的映射。"""
    return {row.path: row.group_id for row in (*task.candidates_a, *task.candidates_b)}


def _folder_candidates(folder: Path, side: str) -> list[Candidate]:
    """读取单类别文件夹的全部图片，附加无真实分组的标识。"""
    if not folder.is_dir():
        raise NotADirectoryError(folder)
    # 类别内的嵌套文件夹仍属于同一类别。
    paths = sorted(p.resolve() for p in folder.rglob('*') if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
    if not paths:
        raise RuntimeError(f"分类文件夹没有图片：{folder}")
    group = f"__folder_{side}__"
    return [Candidate(str(path), group, side) for path in paths]


MANUAL_DISJOINT_POLICY='manual_selected_disjoint'
RANDOM_POLICY='random_support_disjoint'
POLICY_DISPLAY={MANUAL_DISJOINT_POLICY:'手工自选注册',RANDOM_POLICY:'随机注册'}

def resolve_support(task,policy,selected_a,selected_b,random_counts,random_seed):
    """每类按实际类别固定抽样，增加数量保留原排列前缀。"""
    if policy == RANDOM_POLICY:
        result=[]
        for side,count in zip(('A','B'),random_counts):
            pool=sorted(row.path for row in task.candidates(side))
            if not 1 <= count <= len(pool):raise ValueError(f'{side}随机注册数量须在1至{len(pool)}之间')
            identity=getattr(task,'class_'+side.lower()) or '|'.join(pool)
            random.Random(f'{random_seed}:{identity}').shuffle(pool)
            result.append(pool[:count])
        return tuple(result)
    return validate_support(task,selected_a,'A'),validate_support(task,selected_b,'B')

def query_rows(task,policy,support_a,support_b,query_task=None):
    """同池排除注册图，独立待测池要求与注册图互斥。"""
    source=query_task or task
    rows=[(r.path,r.group_id,side) for side in ('A','B') for r in source.candidates(side)]
    selected={str(Path(p).resolve()) for p in support_a+support_b}
    if query_task is not None:
        if selected & {r[0] for r in rows}:raise ValueError('独立待测池与注册图不能重合')
    else:rows=[r for r in rows if r[0] not in selected]
    if not rows:raise ValueError('排除注册图后没有剩余待测图')
    return rows


def _row(value: object) -> dict:
    """统一已导出的字典和会话预测对象。"""
    return asdict(value) if is_dataclass(value) else dict(value) if isinstance(value, dict) else vars(value)


def _ratio(numerator: int, denominator: int) -> float | None:
    """分母为零时明确表示指标不适用。"""
    return numerator / denominator if denominator else None


def class_metrics(rows: Iterable[object], class_a: str, class_b: str,
                  support_counts: tuple[int, int] = (0, 0)) -> dict:
    """按真实类别行、预测类别列汇总；缺真值时不伪报正确率。"""
    records = [_row(row) for row in rows]
    names = [class_a or 'A', class_b or 'B']
    sides = {'A': 0, 'B': 1}
    matrix = [[0, 0], [0, 0]]
    predicted = [0, 0]
    labeled = 0
    for row in records:
        guess, truth = row.get('predicted_side'), row.get('true_side')
        if guess in sides:
            predicted[sides[guess]] += 1
        if truth in sides and guess in sides:
            matrix[sides[truth]][sides[guess]] += 1
            labeled += 1
    # 只有每条预测都有可用真值时才给整体准确率和混淆矩阵。
    complete = bool(records) and labeled == len(records)
    classes = []
    for i, name in enumerate(names):
        count, correct = sum(matrix[i]), matrix[i][i]
        classes.append(dict(name=name, support_count=support_counts[i],
            query_count=count if complete else None, predicted_count=predicted[i],
            correct=correct if complete else None,
            recall=_ratio(correct, count) if complete else None,
            precision=_ratio(correct, predicted[i]) if complete else None))
    correct = matrix[0][0] + matrix[1][1]
    recalls = [item['recall'] for item in classes]
    lower = [name.lower() for name in names]
    ng = lower.index('ng') if 'ng' in lower else None
    ok = lower.index('ok') if 'ok' in lower else None
    return dict(classes=classes, confusion=matrix if complete else None,
        row_column_order=names, query_count=len(records), labeled_count=labeled,
        labels_complete=complete, correct=correct if complete else None,
        accuracy=_ratio(correct, labeled) if complete else None,
        balanced_accuracy=sum(recalls) / 2 if complete and None not in recalls else None,
        ng_miss_rate=(1 - classes[ng]['recall']) if complete and ng is not None and classes[ng]['recall'] is not None else None,
        ok_false_alarm_rate=(1 - classes[ok]['recall']) if complete and ok is not None and classes[ok]['recall'] is not None else None)


def new_session(parent: str | Path) -> Path:
    """按时间与递增序号创建新的评估会话目录。"""
    root = Path(parent).resolve()
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for index in range(1, 1000):
        path = root / f"session_{stamp}_{index:03d}"
        if not path.exists():
            path.mkdir()
            return path
    raise RuntimeError("同一秒内会话超过999个")


def write_result(run: RunResult) -> None:
    """写出会话摘要，并在存在预测时导出完整预测表。"""
    root = Path(run.session_root)
    # 查询池数量按首模型统计一次，避免多模型预测行重复计数。
    first_model = run.metrics[0].model_name if run.metrics else ""
    first_rows = [row for row in run.predictions if row.model_name == first_model] if first_model else []
    summary = {
        "task_id": run.task.task_id,
        "source_mode": run.task.source_mode,
        "source_root": run.task.source_root,
        "class_a": getattr(run.task, "class_a", ""),
        "class_b": getattr(run.task, "class_b", ""),
        "registration_policy": run.policy,
        "support_a": list(run.support_a),
        "support_b": list(run.support_b),
        "candidate_count_a": len(run.task.candidates_a),
        "candidate_count_b": len(run.task.candidates_b),
        "query_count_a": sum(1 for p in first_rows if p.true_side == "A"),
        "query_count_b": sum(1 for p in first_rows if p.true_side == "B"),
        "query_count_after_exclusion": len(first_rows),
        "prediction_rows": len(run.predictions),
        "metrics": [asdict(row) for row in run.metrics],
        "identities": list(run.identities),
        "class_metrics": [{"model_name": metric.model_name,
            **class_metrics([row for row in run.predictions if row.model_name == metric.model_name],
                run.task.class_a, run.task.class_b, (len(run.support_a), len(run.support_b)))}
            for metric in run.metrics],
    }
    (root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    if not run.predictions:
        return
    # 逐图对象先转换为字典，同一数据用于索引和完整 CSV。
    rows = [asdict(row) for row in run.predictions]
    # 索引必须包含模型，避免主／对比模型预测覆盖。
    keyed = {f"{r['model_name']}::{r['task_id']}::{r['query']}": r for r in rows}
    (root / "predictions_index.json").write_text(
        json.dumps(keyed, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    selectable = ['task_id', 'query', 'true_side', 'predicted_side', 'correct',
                  'final_logits_a', 'final_logits_b', 'model_name', 'checkpoint']
    fieldnames = [c for c in selectable if c in rows[0]] + [c for c in rows[0] if c not in selectable]
    with (root / "predictions.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_session_contract(session,*,task,policy,support_a,support_b,queries,checkpoints,query_task=None,extra=None):
    """记录实际输入路径和完整待测分母。"""
    source=query_task or task
    payload=dict(task=dict(task_id=task.task_id,source_root=task.source_root,class_a=task.class_a,class_b=task.class_b),
        policy=policy,support_a=support_a,support_b=support_b,queries=queries,
        checkpoints=[(name,str(path)) for name,path in checkpoints],independent_query_pool=query_task is not None,
        query_source=dict(source_root=source.source_root),query_count=len(queries),**(extra or {}))
    (Path(session)/'session_contract.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
def write_runtime_identities(session,identities):
    """保存模型实际返回的运行身份。"""
    (Path(session)/'runtime_identities.json').write_text(json.dumps(identities,ensure_ascii=False,indent=2),encoding='utf-8')


def format_identity_block(*,code_root,protocol,support_a,support_b,queries,identities=()):
    """将真实运行资源转换为按需展示的文本。"""
    lines=[f'源码：{code_root}',f'注册：{protocol} · A{support_a}/B{support_b} · 待测{queries}']
    for item in identities:
        lines.extend([f"{item.get('model_name','模型')}：{item.get('readout_mode','')} · {item.get('precision','')}",
            f"权重：{item.get('checkpoint','')}",f"适配器：{item.get('adapter_checkpoint','')}"])
    return lines


def make_prediction(*,task,model_name,checkpoint,query,group_id,truth,pred,support_a,support_b,readout_mode,identity,policy):
    """保留真实A/B判断、分数、注册数量与模型资源。"""
    side=pred.predicted_class
    return Prediction(model_name=model_name,checkpoint=str(checkpoint),query=query,group_id=group_id,
        true_side=truth,predicted_side=side,correct=side==truth if truth in ('A','B') else None,
        confidence=float(pred.confidence),probability_a=float(pred.probability_a),probability_b=float(pred.probability_b),
        support_count_a=len(support_a),support_count_b=len(support_b),milliseconds=float(pred.milliseconds),
        task_id=task.task_id,class_a=task.class_a,class_b=task.class_b,source_root=task.source_root,
        readout_mode=readout_mode,registration_policy=policy,precision=identity.get('precision',''),
        adapter_checkpoint=identity.get('adapter_checkpoint',''),final_logits_a=float(pred.final_logits[0]),final_logits_b=float(pred.final_logits[1]))


def run_model(evaluator,create_engine,name,checkpoint,a,b,queries,done,total,progress):
    """每模型一次注册，批量预测保持原顺序和真实完成数。"""
    mode=evaluator.readout_modes[name]
    def report(text):
        """向界面报告真实模型阶段及已完成张数。"""
        if progress:progress(100*done/max(1,total),f'{name} · {text}')
    report('正在载入模型')
    engine=create_engine(checkpoint,device=evaluator.device,support_mode='research',readout_mode=mode,
        official_source=evaluator.official_sources.get(name,''),adapter_checkpoint=evaluator.adapter_checkpoints.get(name,''))
    report('模型已加载')
    try:
        if hasattr(engine,'prepare_task_paths'):engine.prepare_task_paths(a+b+[q[0] for q in queries])
        report(f'正在注册 A{len(a)}/B{len(b)}')
        engine.register_exact_support(a,b,verify_swap=True)
        identity={**engine.describe(),'model_name':name,'checkpoint':str(checkpoint),'readout_mode':mode}
        rows=[]
        for start in range(0,len(queries),32):
            batch=queries[start:start+32]
            predicted=engine.predict([q[0] for q in batch],batch_size=8)
            if len(predicted)!=len(batch):raise RuntimeError('模型返回结果数与待测图片数不一致')
            rows.extend(make_prediction(task=evaluator.task,model_name=name,checkpoint=checkpoint,
                query=q,group_id=g,truth=t,pred=p,support_a=a,support_b=b,readout_mode=mode,
                identity=identity,policy=evaluator.policy) for (q,g,t),p in zip(batch,predicted))
            done+=len(batch);report(f'待测 {done}/{total}')
    finally:
        engine.clear_registration();del engine;report('模型已释放')
    stats=class_metrics(rows,evaluator.task.class_a,evaluator.task.class_b,(len(a),len(b)))
    correct=stats['correct']
    metric=Metrics(name,str(checkpoint),len(rows),correct,len(rows)-correct if correct is not None else None,
        stats['accuracy'],stats['classes'][0]['recall'],stats['classes'][1]['recall'])
    return rows,metric,identity,done


class Evaluator:
    """各模型独立载入、注册和分类，比较共用同一输入池。"""
    def __init__(self,*,task,checkpoints,session_parent,device='cuda',policy=MANUAL_DISJOINT_POLICY,
                 readout_modes=None,official_sources=None,adapter_checkpoints=None,
                 random_counts=(4,4),random_seed=42,query_task=None,fixed_support=None,
                 fixed_queries=None,registration_origin=None):
        """固定用户选择的模型资源、注册策略与可选原待测池。"""
        if policy not in POLICY_DISPLAY:raise ValueError('未知注册方式')
        self.task,self.query_task,self.fixed_queries=task,query_task,fixed_queries
        self.fixed_support=fixed_support
        self.checkpoints=tuple((name,Path(path).resolve()) for name,path in checkpoints if str(path).strip())
        if not self.checkpoints or len({n for n,p in self.checkpoints})!=len(self.checkpoints):raise ValueError('模型名称必须非空且唯一')
        self.session_parent,self.device,self.policy=Path(session_parent),device,policy
        self.readout_modes={n:(readout_modes or {}).get(n,'dinov2_confirmed_support_margin') for n,p in self.checkpoints}
        self.official_sources,self.adapter_checkpoints=dict(official_sources or {}),dict(adapter_checkpoints or {})
        self.random_counts,self.random_seed=random_counts,int(random_seed)
        self.registration_origin=dict(registration_origin or {})

    def run(self,*,selected_a,selected_b,progress=None):
        """整轮共享GPU占用期，全部预测落盘后返回完整结果。"""
        from src.gpu_task import gpu_task
        with gpu_task(self.device):
            if self.fixed_support is None:
                a,b=resolve_support(self.task,self.policy,selected_a,selected_b,self.random_counts,self.random_seed)
            else:
                a,b=([str(Path(p).resolve()) for p in values] for values in self.fixed_support)
                if not a or not b or set(a)&set(b):raise ValueError('注册图片须包含互不重叠的两类')
            queries=list(self.fixed_queries) if self.fixed_queries is not None else query_rows(self.task,self.policy,a,b,self.query_task)
            if not queries:raise ValueError('待测图片不能为空')
            if set(a+b)&{row[0] for row in queries}:raise ValueError('待测图片不能与注册图重合')
            session=new_session(self.session_parent)
            write_session_contract(session,task=self.task,policy=self.policy,support_a=a,support_b=b,
                queries=queries,checkpoints=self.checkpoints,query_task=self.query_task,
                extra=dict(readout_modes=self.readout_modes,official_sources=self.official_sources,
                    adapter_checkpoints=self.adapter_checkpoints,registration_origin=self.registration_origin,
                    random_seed=self.random_seed if self.policy==RANDOM_POLICY else None))
            predictions,metrics,identities=[],[],[]
            done,total=0,len(queries)*len(self.checkpoints)
            for name,checkpoint in self.checkpoints:
                try:rows,metric,identity,done=run_model(self,create_engine,name,checkpoint,a,b,queries,done,total,progress)
                except ValueError as exc:
                    if hasattr(exc,'diagnostics'):
                        (session/'registration_failure.json').write_text(json.dumps(dict(model_name=name,
                            support_a=a,support_b=b,diagnostics=exc.diagnostics),ensure_ascii=False,indent=2),encoding='utf-8')
                    raise
                predictions.extend(rows);metrics.append(metric);identities.append(identity)
            result=RunResult(self.task,self.policy,tuple(a),tuple(b),tuple(metrics),tuple(predictions),str(session),identities=tuple(identities))
            write_result(result);write_runtime_identities(session,identities)
            return result
