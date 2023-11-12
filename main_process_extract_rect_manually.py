import collections
import sys

import cv2
import numpy as np

import train_input
from main_process_video_dump import RESIZE_RATIO_LOW

dataset.forbid_default_video_name()


def main(video_name):
    video_path = dataset.get_video_path(video_name)
    print(f'{video_name=}')
    print(f'{video_path=}')

    cap = cv2.VideoCapture(video_path)
    vw, vh = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(f'{vw=}, {vh=}')

    vals = collections.defaultdict(lambda: None)
    desc = list(
        dict(
            iw='Enter image width(=count of x)',
            ih='Enter image height(=count of y)',
            left='Enter left x of table',
            right='Enter right x of table',
            top='Enter top y of player',
            bottom='Enter bottom y of player'
        ).items()
    )

    def generate_rect():
        try:
            iw, ih, left, right, top, bottom \
                = vals['iw'], vals['ih'], vals['left'], vals['right'], vals['top'], vals['bottom']

            top = top / ih * vh * RESIZE_RATIO_LOW
            bottom = bottom / ih * vh * RESIZE_RATIO_LOW
            left = left / iw * vw * RESIZE_RATIO_LOW
            right = right / iw * vw * RESIZE_RATIO_LOW
            arr = np.array([top, bottom, left, right]).round(0).astype(int)
            a, b, c, d = map(int, arr)
            rect = slice(a, b), slice(c, d)
            return rect
        except (ValueError, ZeroDivisionError, TypeError):
            return None

    def info():
        print('-' * 70)
        for i, (k, v) in enumerate(desc):
            kv = f'{k}={"?" if vals[k] is None else vals[k]}'
            print(f'[{i}] {kv:20s} {v}')

    print('Take a screen shot of a full frame and measure values below.')
    print('Fill all values and enter `done`.')
    info()
    while True:
        print(f'{generate_rect()=}')

        i = input('Input `<index>.<value>` or `info` or `done`? >>> ').strip()
        if i == 'done':
            if any(vals[k] is None for k, _ in desc):
                print('Fill all values!')
                continue
            break
        if i == 'info':
            info()
            continue

        try:
            i, v = map(int, i.split('.'))
            _ = desc[i]
        except (ValueError, IndexError):
            print('Invalid input!')
            continue

        vals[desc[i][0]] = v
        print(f'Set {desc[i][0]}={v}')

    rect = generate_rect()
    print(f'{rect=}')
    r = input('Are you sure to update rect in `./train/rect.json`? [Y/n]: ')
    if r == 'Y':
        train_input.update_rect(video_name, rect)
        print('UPDATED!')
    else:
        print('CANCELED')


if __name__ == '__main__':
    _, *args = sys.argv
    if args:
        vn = args[0]
    else:
        vn = None
    assert vn is not None, 'video_name=None is forbidden'
    main(video_name=vn)
