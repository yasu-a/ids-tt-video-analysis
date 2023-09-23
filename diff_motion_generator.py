import glob
import os

import cv2

from extractimio import VideoFrameIO, MultiprocessingVideoFrameIO

path = os.path.expanduser(r'~\Desktop/idsttvideos/singles')
glob_pattern = os.path.join(path, r'**/*.mp4')
video_path = glob.glob(glob_pattern, recursive=True)[0]
print(video_path)

cache_base_path = os.path.expanduser(r'~\Desktop/vio_cache')


def main():
    with VideoFrameIO(video_path, cache_base_path=cache_base_path) as vio:
        def pair_producer(it):
            prev = None
            for item in it:
                if prev is not None:
                    yield prev, item
                prev = item

        import numpy as np
        from tqdm import tqdm

        def mapper_gauss(frame):
            return cv2.GaussianBlur(frame, (51, 51), 100)

        def frame_map(fn, it):
            for i, frame in it:
                yield i, fn(frame)

        def delta_producer(it, formula):
            for (ix, fx), (iy, fy) in pair_producer(it):
                f_delta = formula(fx, fy)
                yield iy, f_delta

        STEP = 5
        START = 0
        END = 40000
        it_src = vio.loc[START:END:STEP]
        it = vio.loc[START:END:STEP]
        it = frame_map(mapper_gauss, it)
        it = delta_producer(it, formula=lambda fx, fy: np.square((fy - fx) / 256.0))
        it = pair_producer(it)
        it = tqdm(it, total=(END - START) // STEP)

        t = []

        tot = []

        for i, (((_, f_delta_prev), (i_frame, f_delta)), (_, f_src)) in enumerate(zip(it, it_src)):
            t.append(vio.timestamp(i_frame))

            y = f_delta * f_delta_prev
            y = (y * 256.0).astype(np.uint8)

            tot.append(y.mean(axis=2).mean(axis=0))

            F_STEP = 4
            h = f_src[::F_STEP, ::F_STEP].shape[0]
            tot_vs = np.clip(np.vstack(tot) * 3, 0, 255)[::1, ::10]
            graph = np.zeros((h, tot_vs.shape[1], 3), dtype=np.uint8)
            tot_vs = tot_vs[-h:, :]
            graph[:tot_vs.shape[0], :tot_vs.shape[1], :] = np.dstack([tot_vs[:tot_vs.shape[0], :tot_vs.shape[1]]] * 3)
            graph[:, :, [0, 2]] = 0
            graph[tot_vs.shape[0] - 1, :, 0] = 255

            if i % 10 == 0:
                np.savez('diff_out.npz', np.vstack(tot), t)


if __name__ == '__main__':
    main()
