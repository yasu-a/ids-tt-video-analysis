import os

import cv2
import imageio.v2 as iio

from .common import Locator
from .diskio import VideoFrameCache


class VideoFrameIO:
    def __init__(self, video_path, cache_base_path=None):
        cache_base_path = cache_base_path or './vio_cache'

        os.makedirs(cache_base_path, exist_ok=True)

        self.__video_path = video_path

        _, video_name = os.path.split(video_path)
        cache_path = os.path.join(cache_base_path, video_name + '.zip')
        self.__cache = VideoFrameCache(cache_path)

        self.__reader = None
        self.__props = {}

        self.loc = Locator(self)

    def __len__(self):
        return self.num_frames

    def __read_frame(self, frame_index):
        frame = self.__reader.get_data(frame_index)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        return frame

    def __getitem__(self, frame_index):
        if frame_index < 0 or self.num_frames <= frame_index:
            raise IndexError()

        frame = self.__cache.get_frame(frame_index)
        if frame is None:
            frame = self.__read_frame(frame_index)
            self.__cache.put_frame(frame_index, frame)
        return frame

    def pos(self, frame_index):
        return frame_index / self.fps

    @property
    def fps(self):
        return self.__props['fps']

    @property
    def num_frames(self):
        return self.__props['num_frames']

    def _start(self):
        self.__reader = iio.get_reader(self.__video_path)

        self.__props['fps'] = self.__reader.get_meta_data()['fps']

        video = cv2.VideoCapture(self.__video_path)
        self.__props['num_frames'] = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        del video

        self.__cache.open()

    def __enter__(self):
        self._start()
        return self

    def _close(self):
        self.__cache.close()
        self.__reader.close()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._close()
        return False
