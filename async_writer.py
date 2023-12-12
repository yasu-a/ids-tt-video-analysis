import multiprocessing as mp
import os

import imageio.v3 as iio
import numpy as np

import app_logging

MAX_BACKLOG_FRAMES = 128

logger = app_logging.create_logger(__name__)


def _worker(q: mp.Queue, params: dict):
    # FIXME: stopping me sometimes requires process kill

    with iio.imopen(params['path'], "w", plugin="pyav") as out:
        logger.info(f'Worker started: {out}')
        out.init_video_stream("vp9", fps=params['fps'])

        while True:
            frame = q.get(block=True)
            if frame is None:
                break
            if frame.ndim == 2:
                frame = np.repeat(frame[..., None], repeats=3, axis=2)
            if not (frame.ndim == 3 and frame.shape[2] == 3):
                logger.error(f'invalid frame shape {frame.shape}')
                break
            out.write_frame(frame)

    logger.info('Worker closed')


class AsyncVideoFrameWriter:
    def __init__(self, path, fps):
        path = os.path.abspath(path)
        os.makedirs(os.path.dirname(path), exist_ok=True)

        logger.info(f'{path=}, {fps=}')

        self.__q = mp.Queue(maxsize=MAX_BACKLOG_FRAMES)
        params = dict(path=path, fps=fps)
        self.__p = mp.Process(target=_worker, args=(self.__q, params))

    def write(self, frame):
        if not self.__p.is_alive():
            raise ValueError('writer not open')
        self.__q.put(frame)

    def __enter__(self):
        logger.info('Enter')
        self.__p.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        logger.info('Exit')
        self.__q.put(None)
        self.__p.join()
        logger.info('Joined')
        return False
