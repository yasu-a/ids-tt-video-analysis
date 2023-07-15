import timeit
import numpy as np
import pandas as pd


class Timer:
    def __init__(self, *args, **kwargs):
        assert len(args) == 0 and len(kwargs) == 0

    def setup(self):
        pass

    def target(self):
        pass

    def finalize(self):
        pass

    @classmethod
    def measure(cls, *args, _repeat=None, _number=None, **kwargs):
        job = cls(*args, **kwargs)
        job.setup()
        measure(job.target, repeat=_repeat, number=_number)
        job.finalize()


def measure(f, repeat=None, number=None):
    repeat = repeat or 5
    number = number or 1
    times = timeit.repeat(f, repeat=repeat, number=number)
    df = pd.DataFrame(
        {'num': np.arange(repeat) + 1, 'time': times, 'loop': np.array(times) / number})
    desc = df.describe().loop
    print(f'{desc["mean"] * 1000:8.3f}ms ± {desc["std"] * 1000 * 2:8.3f}ms (CI=95%)')
