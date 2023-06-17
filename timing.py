import time

from tqdm import tqdm
import glob
import os
import timeit

import cv2
import numpy as np
import pandas as pd

from extractimio import VideoFrameIO, MultiprocessingVideoFrameIO

path = os.path.expanduser(r'~\Desktop/idsttvideos/singles')
glob_pattern = os.path.join(path, r'**/*.mp4')
video_path = glob.glob(glob_pattern, recursive=True)[0]
print(video_path)

cache_base_path = os.path.expanduser(r'~\Desktop/vio_cache')


def measure(f):
    repeat = 3
    number = 1
    times = timeit.repeat(f, repeat=repeat, number=number)
    df = pd.DataFrame({'num': np.arange(repeat) + 1, 'time': times, 'loop': np.array(times) / number})
    desc = df.describe().loop
    print(f'{desc["mean"] * 1000:8.3f}ms ± {desc["std"] * 1000 * 2:8.3f}ms (CI=95%)')


def main():
    import imageio.v2 as iio

    STEP = 1

    with VideoFrameIO(video_path, cache_base_path=cache_base_path) as vio:
        def fn():
            for frame_index, frame in tqdm(vio.loc[:30]):
                time.sleep(0.2)

        measure(fn)

    with MultiprocessingVideoFrameIO(video_path, cache_base_path=cache_base_path) as vio:
        def fn():
            for frame_index, frame in tqdm(vio.loc[:30]):
                time.sleep(0.2)

        measure(fn)


if __name__ == '__main__':
    main()
