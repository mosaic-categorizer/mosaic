import os
import signal
import sys
import time
from concurrent.futures import ProcessPoolExecutor, CancelledError
from concurrent.futures.process import BrokenProcessPool
from copy import copy
from time import sleep

from tqdm import tqdm


class ProcessPool:

    def __init__(self, size: int):
        self._executor = ProcessPoolExecutor(max_workers=size)
        self._size = size
        self._futures = []
        self._to_process = []
        self._fn = None
        self._args = None
        self._queue_size = None

    def submit(self, fn, *args) -> None:
        self._futures.append(self._executor.submit(fn, *args))
        if len(self._futures) <= self._size and len(self._to_process) % 5 == 0:
            sleep(10)

    def batch_submit(self, tasks: list, fn, *args, queue_size: int = 1024) -> None:
        self._to_process = copy(tasks)
        self._fn = fn
        self._args = args
        self._queue_size = queue_size
        self.submit_more_tasks(0)

    def submit_more_tasks(self, n_done: int):
        if self._queue_size is None:
            return
        n_running = len(self._futures) - n_done
        for _ in range(self._queue_size - n_running):
            if len(self._to_process) == 0:
                return
            args = [self._to_process.pop(0)]
            args.extend(self._args)
            self.submit(self._fn, *args)

    def get_n_done(self) -> int:
        return sum([1 if f.done() and not f.cancelled() else 0 for f in self._futures])

    def get_result(self) -> list:
        results = []
        for f in self._futures:
            try:
                results.append(f.result())
            except (BrokenProcessPool, CancelledError):
                pass
        return results

    def kill(self) -> None:
        if not self._executor._processes:
            raise RuntimeError("No processes running")
        for pid in self._executor._processes:
            os.kill(pid, signal.SIGKILL)
        self._executor.shutdown(wait=True, cancel_futures=True)

    def is_running(self) -> bool:
        if not self._executor._processes:
            return False
        return len(self._to_process) > 0 or sum(
            [1 if f.done() else 0 for f in self._futures]
        ) < len(self._futures)

    def wait_completion(self, unit: str = "traces", timeout: int = -1) -> None:
        start_time = time.time()
        with tqdm(
            total=len(self._futures) + len(self._to_process), file=sys.stdout, unit=unit
        ) as pbar:
            last_count = 0
            while self.is_running():
                time.sleep(1)
                count = self.get_n_done()
                self.submit_more_tasks(count)
                if count > last_count:
                    pbar.update(count - last_count)
                    last_count = count
                else:
                    pbar.refresh()
                if 0 < timeout < (time.time() - start_time):
                    self.kill()
