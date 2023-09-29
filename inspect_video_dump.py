import sys

import numpy as np

np.set_printoptions(suppress=True)

import dataset


def inspect(video_name, rect, start=None, stop=None):
    assert start is None, 'providing start is forbidden'

    with dataset.VideoFrameStorage(
            dataset.get_video_frame_dump_dir_path(video_name),
            mode='r'
    ) as vf_store:
        print(f'output={vf_store.count()}')


def main(video_name):
    rect = slice(70, 260), slice(180, 255)  # height, width
    # height: 奥の選手の頭から手前の選手の足がすっぽり入るように
    # width: ネットの部分の卓球台の幅に合うように

    inspect(video_name, rect)


if __name__ == '__main__':
    _, *args = sys.argv
    if args:
        video_name = args[0]
    else:
        video_name = None
    main(video_name=video_name)
