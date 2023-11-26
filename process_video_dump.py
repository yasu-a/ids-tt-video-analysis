import argparse
import datetime
import os

import cv2
import numpy as np
from tqdm import tqdm

import app_logging
import npstorage_context as snp_context
import process
import storage
import storage.npstorage as snp
from config import config

if __name__ == '__main__':
    app_logging.set_log_level(app_logging.DEBUG)
    config.enable_debug_mode()


class ProcessStageVideoDump(process.ProcessStage):
    NAME = 'dump-video'
    ALIASES = 'dv',

    @classmethod
    def customize_parser(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument('video_name', type=str)
        parser.add_argument('-r', '--resize-ratio', type=float)
        parser.add_argument('-s', '--step', type=int)
        parser.add_argument('-d', '--diff-luminance-scale', '--scale', type=float)

    def __init__(
            self,
            video_name: str,
            resize_ratio: float = 0.3,
            step: int = 5,
            diff_luminance_scale: float = 5.0
    ):
        self.__video_name = video_name
        self.__resize_ratio = resize_ratio
        self.__step = step
        self.__diff_luminance_scale = diff_luminance_scale

        self.__video_path = os.path.join(config.video_location, video_name + '.mp4')

        self.__cap = None
        self.__frame_count = None

    def _init_video_capture(self):
        self.__cap = cv2.VideoCapture(self.__video_path)
        self.__frame_count = int(self.__cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def _iter_flags(self):
        pattern = np.zeros(self.__step, dtype=int)
        pattern[:3] = [1, 2, 3]

        for fi in range(self.__frame_count):
            flag = pattern[fi % self.__step]
            yield fi, flag

    def _iter_frames(self, fi_flag_pair_lst) -> tuple:  # fi, ts, stack
        def retrieve_frame():
            _, image = self.__cap.retrieve()
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return image

        self.__cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        bar = tqdm(fi_flag_pair_lst)
        stack = []
        yield_count = 0
        for fi, flag in bar:
            if not self.__cap.grab():
                break

            ts = self.__cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            bar.set_description(f'{datetime.timedelta(seconds=ts)!s} {yield_count:5}')

            if len(stack) == flag - 1:
                stack.append(retrieve_frame())
                if flag == 3:
                    yield fi, ts, tuple(stack)
                    stack.clear()

    @classmethod
    def _to_uint8(cls, a):
        return np.clip((a * 256.0).astype(int), 0, 255).astype(np.uint8)

    def run(self):
        self._init_video_capture()

        fi_flag_pair_lst = list(self._iter_flags())
        n_output = int(sum(flag == 3 for idx, flag in fi_flag_pair_lst))

        with storage.create_instance(
                domain='numpy_storage',
                entity=self.__video_name,
                context='frames',
                mode='w',
                n_entries=n_output
        ) as snp_video_frame:
            assert isinstance(snp_video_frame, snp.NumpyStorage)

            for j, (fi, timestamp, images) \
                    in enumerate(self._iter_frames(fi_flag_pair_lst)):
                images = list(images)

                # preprocess
                for i in range(len(images)):
                    im = images[i]
                    im = cv2.resize(im, None, fx=self.__resize_ratio, fy=self.__resize_ratio)
                    im = im.astype(np.float32)
                    im /= 256.0
                    # noinspection PyArgumentList
                    assert im.max() < 1.0, im.max()
                    images[i] = im

                # copy original
                _, original, _ = images
                original = original.copy()

                # generate diff
                x, y, z = images
                diff_img = np.sqrt(np.sqrt(np.square(y - x) * np.square(z - y)))
                diff_img_gaussian = cv2.GaussianBlur(diff_img, (3, 3), 1)
                motion = np.clip(diff_img_gaussian * self.__diff_luminance_scale, 0.0, 1.0 - 1e-6)

                # dump
                snp_video_frame[j] = snp_context.SNPEntryVideoFrame(
                    original=self._to_uint8(original),
                    motion=self._to_uint8(motion),
                    timestamp=timestamp,
                    fi=fi
                )


if __name__ == '__main__':
    ProcessStageVideoDump('20230205_04_Narumoto_Harimoto').run()
