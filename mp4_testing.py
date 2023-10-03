import numpy as np

import async_writer

W, H = 100, 200
F = 100
FPS = 30


def iter_frames():
    for i in range(F):
        fr = np.full(shape=(W, H), fill_value=i % 256)
        yield i, fr


if __name__ == '__main__':
    with async_writer.AsyncVideoFrameWriter('out.mp4', FPS) as wr:
        for i, fr in iter_frames():
            wr.write(fr)
