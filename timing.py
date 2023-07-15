import time
import timeit
import numpy as np
import pandas as pd


class Timer:
    def __init__(self, *args, **kwargs):
        assert len(args) == 0 and len(kwargs) == 0

    def setup(self):
        pass

    def context(self):
        return None

    def target(self):
        pass

    def finalize(self):
        pass

    @classmethod
    def measure(cls, *args, _repeat=5, **kwargs):
        times = []

        for _ in range(_repeat):
            job = cls(*args, **kwargs)
            job.setup()

            start, end = None, None

            def run():
                nonlocal start, end
                start = time.perf_counter()
                job.target()
                end = time.perf_counter()

            ctx = job.context()
            if ctx is not None:
                with ctx:
                    run()
            else:
                run()

            job.finalize()

            times.append(end - start)

        times = np.array(times)
        mean = times.mean()
        se = np.std(times, ddof=1) / np.sqrt(len(times))
        print(f'{mean * 1000:8.3f}ms ± {se * 1000 * 2:8.3f}ms (CI=95%)')


def measure(f, repeat=None, number=None):
    repeat = repeat or 5
    number = number or 1
    times = timeit.repeat(f, repeat=repeat, number=number)
    df = pd.DataFrame(
        {'num': np.arange(repeat) + 1, 'time': times, 'loop': np.array(times) / number})
    desc = df.describe().loop
    print(f'{desc["mean"] * 1000:8.3f}ms ± {desc["std"] * 1000 * 2:8.3f}ms (CI=95%)')
