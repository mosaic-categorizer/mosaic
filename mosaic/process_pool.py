import os
import signal
import sys
import time
from concurrent.futures import ProcessPoolExecutor, CancelledError
from concurrent.futures.process import BrokenProcessPool

from tqdm import tqdm


class ProcessPool:

    def __init__(self, size: int):
        self._executor = ProcessPoolExecutor(max_workers=size)
        self._futures = []

    def submit(self, fn, *args) -> None:
        self._futures.append(self._executor.submit(fn, *args))

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
        return sum([1 if f.done() else 0 for f in self._futures]) < len(self._futures)

    def wait_completion(self, unit: str = 'traces', timeout: int = -1) -> None:
        start_time = time.time()
        with tqdm(total=len(self._futures), file=sys.stdout, unit=unit) as pbar:
            last_count = 0
            while self.is_running():
                time.sleep(1)
                count = self.get_n_done()
                if count > last_count:
                    pbar.update(count - last_count)
                    last_count = count
                else:
                    pbar.refresh()
                if 0 < timeout < (time.time() - start_time):
                    self.kill()