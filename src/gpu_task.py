"""本模块协调 Windows 上的训练与推理 CUDA 任务。
经过测试我们发现，跑训练和推理时可以通过调度gpu task来提高效率，避免进程死锁或是GPU占用率过低/显存占用过高
调用方通过上下文持有单 GPU 锁；维护脚本入口也在同一子进程内持锁执行。"""
from contextlib import contextmanager
from pathlib import Path
import tempfile
import threading
import sys

# -m入口和脚本内导入共享同一线程嵌套状态。
if __name__ == '__main__':
    sys.modules['src.gpu_task'] = sys.modules[__name__]

LOCK_PATH = Path(tempfile.gettempdir()) / 'support_conditioned_vision_single_gpu.lock'
_THREAD_GATE = threading.Lock()
_CONTEXT = threading.local()


class GpuTaskBusy(RuntimeError):
    """已有 CUDA 任务占用单卡时明确拒绝新任务。"""

@contextmanager
def gpu_task(device: str):
    """CUDA 任务持有同一字节锁；CPU 直接执行，异常自动释放。"""
    if str(device) == 'auto':
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if not str(device).startswith('cuda'):
        yield
        return
    # 同线程嵌套复用已有锁，最外层退出时才释放跨进程锁。
    if getattr(_CONTEXT, 'depth', 0):
        _CONTEXT.depth += 1
        try:
            yield
        finally:
            _CONTEXT.depth -= 1
        return
    # 先限制本进程并发线程，再用文件字节锁协调其他进程。
    if not _THREAD_GATE.acquire(blocking=False):
        raise GpuTaskBusy('已有训练或推理任务占用 GPU，请等待当前任务结束。')
    import msvcrt
    try:
        with LOCK_PATH.open('a+b') as stream:
            # 确保锁定的首字节存在，并将锁范围定位到该字节。
            stream.seek(0, 2)
            if stream.tell() == 0:
                stream.write(b'0')
                stream.flush()
            stream.seek(0)
            try:
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as error:
                raise GpuTaskBusy('已有训练或推理任务占用 GPU，请等待当前任务结束。') from error
            _CONTEXT.depth = 1
            try:
                yield
            finally:
                _CONTEXT.depth = 0
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
    finally:
        _THREAD_GATE.release()

def main(argv=None):
    """同一真实子进程持锁执行原维护脚本，保留原参数和兄弟模块导入。"""
    import argparse
    import runpy
    parser = argparse.ArgumentParser(description='串行执行原SCT GPU维护脚本')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('script')
    parser.add_argument('arguments', nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    script = Path(args.script).resolve()
    sys.argv = [str(script), *args.arguments]
    sys.path[0] = str(script.parent)
    with gpu_task(args.device):
        runpy.run_path(str(script), run_name='__main__')
    return 0

if __name__ == '__main__':
    raise SystemExit(main())