import multiprocessing as mp
import os
import time

import imageio.v2 as iio

MAX_BACKLOG_FRAMES = 128


def _worker(q: mp.Queue, params: dict):
    out = iio.get_writer(
        params['path'],
        format='FFMPEG',
        fps=params['fps'],
        mode='I',
        codec='h264',
        quality=4
    )

    while True:
        if q.empty():
            time.sleep(0.1)
        else:
            frame = q.get(block=False)
            if frame is None:
                break
            out.append_data(frame)

    out.close()


class AsyncVideoFrameWriter:
    def __init__(self, path, fps):
        path = os.path.abspath(path)
        os.makedirs(os.path.dirname(path), exist_ok=True)

        self.__q = mp.Queue(maxsize=MAX_BACKLOG_FRAMES)
        params = dict(path=path, fps=fps)
        self.__p = mp.Process(target=_worker, args=(self.__q, params))

    def write(self, frame):
        if not self.__p.is_alive():
            raise ValueError('writer not open')
        self.__q.put(frame)

    def __enter__(self):
        self.__p.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.__q.put(None)
        self.__p.join()
        return False
