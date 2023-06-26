import timeit
import numpy as np
import pandas as pd


def measure(f):
    repeat = 5
    number = 1
    times = timeit.repeat(f, repeat=repeat, number=number)
    df = pd.DataFrame(
        {'num': np.arange(repeat) + 1, 'time': times, 'loop': np.array(times) / number})
    desc = df.describe().loop
    print(f'{desc["mean"] * 1000:8.3f}ms ± {desc["std"] * 1000 * 2:8.3f}ms (CI=95%)')
