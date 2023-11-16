import sys

import numpy as np

import train_input

np.set_printoptions(suppress=True)

from tqdm import tqdm

from util_extrema_feature_motion_detector import ExtremaFeatureMotionDetector

import dataset

dataset.forbid_writing = True

def process_rect(rect):
    w = rect[1].stop - rect[1].start
    aw = int(w * 1.0)
    return slice(rect[0].start, rect[0].stop), slice(rect[1].start - aw, rect[1].stop + aw)


NUM_MOTION_VECTORS_MAX = 64


def process(video_name, rect=None, start=None, stop=None):
    if rect is None:
        rect = train_input.load_rect(video_name)

    with dataset.VideoFrameStorage(
            dataset.get_video_frame_dump_dir_path(video_name),
            mode='r'
    ) as vf_store:
        rect = process_rect(rect)
        start = start or 0
        stop = stop or vf_store.count() - 1

        detector = ExtremaFeatureMotionDetector(
            rect
        )

        for i in tqdm(range(start, stop)):
            src_current, src_next = vf_store.get(i), vf_store.get(i + 1)

            motion_images = src_current['motion'], src_next['motion']
            original_images = src_current['original'], src_next['original']
            ts = src_current['timestamp']

            result = detector.compute(original_images, motion_images)

            def process_matrix(m):
                if m is None:
                    return np.full((NUM_MOTION_VECTORS_MAX, 2), np.nan)
                m = m.astype(np.float32)
                n_pad = NUM_MOTION_VECTORS_MAX - m.shape[0]
                pad = np.full((n_pad, 2), np.nan)
                m = np.concatenate([m, pad], axis=0)
                assert len(m) == NUM_MOTION_VECTORS_MAX, m.shape
                return m

            if result['valid']:
                data_dct = dict(
                    start=process_matrix(result['global_motion_center_a']),
                    end=process_matrix(result['global_motion_center_b']),
                    frame_index=i,
                    timestamp=ts
                )
            else:
                data_dct = dict(
                    start=process_matrix(None),
                    end=process_matrix(None),
                    frame_index=i,
                    timestamp=ts
                )


def main(video_name):
    rect = slice(70, 260), slice(180, 255)  # height, width
    # height: 奥の選手の頭から手前の選手の足がすっぽり入るように
    # width: ネットの部分の卓球台の幅に合うように

    process(video_name, rect, start=200, stop=201)


if __name__ == '__main__':
    _, *args = sys.argv
    if args:
        video_name = args[0]
    else:
        video_name = None
    main(video_name=video_name)
