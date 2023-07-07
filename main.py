import glob
import os

import cv2

from extract import VideoFrameReader

path = os.path.expanduser(r'~\Desktop/idsttvideos/singles')
glob_pattern = os.path.join(path, r'**/*.mp4')
video_path = glob.glob(glob_pattern, recursive=True)[0]
print(video_path)

cache_base_path = os.path.expanduser(r'~\Desktop/vio_cache')


def main():
    vfr = VideoFrameReader(video_path)

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

    STEP = 2
    START = 0
    END = 60

    it_src = vfr[START:END:STEP]

    it = vfr[START:END:STEP]
    it = frame_map(mapper_gauss, it)
    it = delta_producer(it, formula=lambda fx, fy: np.square((fy - fx) / 256.0))
    it = pair_producer(it)
    it = tqdm(it, total=(END - START) // STEP)

    fds = []
    fs = []
    t = []

    import imageio.v2 as iio

    out = iio.get_writer(
        'out.mp4',
        format='FFMPEG',
        fps=vfr.fps / STEP,
        mode='I',
        codec='h264',
    )

    tot = []

    for i, (((_, f_delta_prev), (frame_time, f_delta)), (_, f_src)) in enumerate(zip(it, it_src)):
        t.append(frame_time)

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
        out.append_data(
            cv2.cvtColor(
                np.concatenate(
                    [
                        f_src[::F_STEP, ::F_STEP],
                        y[::F_STEP, ::F_STEP],
                        graph
                    ],
                    axis=1
                ),
                cv2.COLOR_BGR2RGB)
        )

        if i % 10 == 0:
            np.save('out.npy', np.vstack(tot))

    out.close()


if __name__ == '__main__':
    main()
