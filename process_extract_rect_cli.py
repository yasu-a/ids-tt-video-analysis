import argparse
import collections
import os

import numpy as np

import process
import train_input
from config import config


class ProcessStageExtractRectCLI(process.ProcessStage):
    NAME = 'extract-rect-cli'
    ALIASES = 'cli-rect',

    @classmethod
    def customize_parser(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument('video_name', type=str)

    def __init__(self, video_name):
        self.__video_name = video_name
        self.__video_path = os.path.join(config.video_location, video_name)

        self.__vals = collections.defaultdict(lambda: None)
        self.__desc = list(
            dict(
                iw='Enter image width(=count of x)',
                ih='Enter image height(=count of y)',
                left='Enter left x of table',
                right='Enter right x of table',
                top='Enter top y of player',
                bottom='Enter bottom y of player'
            ).items()
        )

        self.__preview_enabled = False

    # def _get_size(self) -> tuple[float, float]:
    #     cap = cv2.VideoCapture(self.__video_path)
    #     vh, vw = cap.get(cv2.CAP_PROP_FRAME_HEIGHT), cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    #     return vh, vw

    def _generate_rect(self):
        try:
            vals = self.__vals

            iw, ih, left, right, top, bottom = (
                vals['iw'], vals['ih'], vals['left'], vals['right'], vals['top'], vals['bottom']
            )

            # noinspection PyUnresolvedReferences
            top, bottom = top / ih, bottom / ih
            # noinspection PyUnresolvedReferences
            left, right = left / iw, right / iw
            rect = (top, bottom), (left, right)
            return rect
        except (ValueError, ZeroDivisionError, TypeError):
            return None

    def __info(self):
        print('-' * 70)
        for i, (k, v) in enumerate(self.__desc):
            kv = f'{k}={"?" if self.__vals[k] is None else self.__vals[k]}'
            print(f'[{i}] {kv:20s} {v}')

    # FIXME: preview() is not working
    #  (-215:Assertion failed) !_src.empty() in function 'cv::cvtColor'
    #  File "F:\マイドライブ\PycharmProjects\iDSTTVideoAnalysis\process_extract_rect_cli.py",
    #  line 87, in __preview
    def __preview(self):
        if not self.__preview_enabled:
            return

        import cv2
        import storage
        import storage.npstorage as snp

        with storage.create_instance(
                domain='numpy_storage',
                entity=self.__video_name,
                context='frames',
                mode='r',
        ) as snp_video_frame:
            assert isinstance(snp_video_frame, snp.NumpyStorage)
            random_entry = snp_video_frame.get_entry(np.random.randint(snp_video_frame.count()))
            image = random_entry.original

        cv2.imshow('winname', image)

    def run(self):
        # DEBUG:
        # a = [1746, 982, 715, 1020, 200, 781]
        # for i in range(6):
        #     self.__vals[self.__desc[i][0]] = a[i]

        print(f'{self.__video_name=}')
        print(f'{self.__video_path=}')

        print('Take a screen shot of a full frame and measure values below.')
        print('Fill all values and enter `done`.')
        self.__info()
        while True:
            print(f'rect={self._generate_rect()}')
            self.__preview()

            i = input('Input `<index>.<value>`, `info`, `preview`, or `done`? >>> ').strip()
            if i == 'done':
                if any(self.__vals[k] is None for k, _ in self.__desc):
                    print('Fill all values!')
                    continue
                break
            if i == 'info':
                self.__info()
                continue
            if i == 'preview':
                self.__preview_enabled = True
                continue

            try:
                i, v = map(int, i.split('.'))
                _ = self.__desc[i]
            except (ValueError, IndexError):
                print('Invalid input!')
                continue

            self.__vals[self.__desc[i][0]] = v
            print(f'Set {self.__desc[i][0]}={v}')

        rect = self._generate_rect()
        print(f'{rect=}')
        r = input('Are you sure to update rect in `./train/rect.json`? [Y/n]: ')
        if r == 'Y':
            train_input.update_rect(self.__video_name, rect)
            print('UPDATED!')
        else:
            print('CANCELED')
