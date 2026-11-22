"""验证固定 Support 下带 GT 的 Query，输出逐图预测与二分类指标。"""
import json
from infer import build_parser, run


def main(argv=None):
    """要求真实 Query 的 GT，并运行固定 Support 注册、预测和指标导出。"""
    args = build_parser(validation=True).parse_args(argv)
    result = run(args)
    print(json.dumps({
        'model_version': result['model_version'],
        'queries': result['queries'],
        'science_status': result['science_status'],
        'swap_verification': result['swap_verification'],
        'metrics': result['metrics'],
    }, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
