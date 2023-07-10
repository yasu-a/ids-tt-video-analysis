import multiprocessing as mp
import time
import imageio as iio


class AsyncVideoFrameWriter:
    def __init__(self, path, fps):
        self.__q = mp.Queue()
        params = dict(path=path, fps=fps)
        self.__p = mp.Process(target=self.__worker, args=(self.__q, params))

    @classmethod
    def __worker(cls, q: mp.Queue, params: dict):
        out = iio.get_writer(
            params['path'],
            format='FFMPEG',
            fps=params['fps'],
            mode='I',
            codec='h264',
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

    def write(self, frame):
        self.__q.put(frame)

    def __enter__(self):
        self.__p.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.__q.put(None)
        self.__p.join()
        return False
