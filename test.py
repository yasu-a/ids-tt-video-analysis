import sys

import numpy as np

import train_input

np.set_printoptions(suppress=True)

import matplotlib.pyplot as plt
import dataset


def process_rect(rect):
    w = rect[1].stop - rect[1].start
    aw = int(w * 1.0)
    return slice(rect[0].start, rect[0].stop), slice(rect[1].start - aw, rect[1].stop + aw)


NUM_MOTION_VECTORS_MAX = 64


def process(video_name, rect=None, start=None, stop=None):
    assert start is None, 'providing start is forbidden'

    if rect is None:
        rect = train_input.load_rect(video_name)

    with dataset.VideoFrameStorage(
            dataset.get_video_frame_dump_dir_path(video_name),
            mode='r'
    ) as vf_store:
        for i in range(200, 210):
            data = vf_store.get(i)
            plt.figure()
            plt.imshow(data['motion'])
            plt.show()


def main(video_name):
    rect = slice(70, 260), slice(180, 255)  # height, width
    # height: 奥の選手の頭から手前の選手の足がすっぽり入るように
    # width: ネットの部分の卓球台の幅に合うように

    process(video_name, rect)


if __name__ == '__main__':
    _, *args = sys.argv
    if args:
        video_name = args[0]
    else:
        video_name = None
    main(video_name=video_name)
