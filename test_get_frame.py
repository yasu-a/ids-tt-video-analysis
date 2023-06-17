import glob
import os
import timeit

import numpy as np
import pandas as pd
from tqdm import tqdm

from extractimio import MultiprocessingVideoFrameIO

path = os.path.expanduser(r'~\Desktop/idsttvideos/singles')
glob_pattern = os.path.join(path, r'**/*.mp4')
video_path = glob.glob(glob_pattern, recursive=True)[0]
print(video_path)

cache_base_path = os.path.expanduser(r'~\Desktop/vio_cache')


def measure(f):
    repeat = 5
    number = 3
    times = timeit.repeat(f, repeat=repeat, number=number)
    df = pd.DataFrame({'num': np.arange(repeat) + 1, 'time': times, 'loop': np.array(times) / number})
    desc = df.describe().loop
    print(f'{desc["mean"] * 1000:8.3f}ms ± {desc["std"] * 1000 * 2:8.3f}ms (CI=95%)')


def main():
    with MultiprocessingVideoFrameIO(video_path, cache_base_path=cache_base_path) as vio:
        import imageio.v2 as iio

        out = iio.get_writer(
            'out.mp4',
            format='FFMPEG',
            fps=vio.fps,
            mode='I',
            codec='h264',
        )

        bar = tqdm(vio.loc[:120])
        for frame_index, frame in bar:
            bar.set_description(
                ' '.join(map(str, [frame_index, frame.shape]))
            )
            out.append_data(frame)

        out.close()


if __name__ == '__main__':
    main()
