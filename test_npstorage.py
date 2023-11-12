import datetime
import os

import cv2
import numpy as np
from tqdm import tqdm

import app_logging
import npstorage_context as snp_context
import storage.npstorage as snp
from config import config

app_logging.set_log_level(app_logging.DEBUG)

if __name__ == '__main__':
    video_name = '20230205_04_Narumoto_Harimoto'
    video_path = os.path.join(config.video_location, video_name + '.mp4')
    RESIZE_RATIO = 0.3
    STEP = 5

    # create VideoCapture instance
    cap = cv2.VideoCapture(video_path)

    # extract meta
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_rate = float(cap.get(cv2.CAP_PROP_FPS))


    # output frame index and role
    def flag_it():
        pattern = np.zeros(STEP, dtype=int)
        pattern[:3] = [1, 2, 3]

        for index in range(frame_count):
            flag = pattern[index % STEP]
            yield index, flag


    index_flag_pairs = list(flag_it())

    # calc n_output
    n_output = int(sum(f == 3 for idx, f in index_flag_pairs))

    # iter frames
    stack = []


    def iter_frames() -> tuple:
        def retrieve_frame():
            _, image = cap.retrieve()
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return image

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        yield_count = 0
        bar = tqdm(index_flag_pairs)
        for index, flag in bar:
            if not cap.grab():
                break

            ts = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            bar.set_description(f'{datetime.timedelta(seconds=ts)!s} {yield_count:5}')

            if len(stack) == flag - 1:
                stack.append(retrieve_frame())
                if flag == 3:
                    yield ts, tuple(stack)
                    stack.clear()


    with snp.create_instance(
            entity=video_name,
            context='frames',
            mode='w',
            n_entries=n_output
    ) as snp_video_frame:
        snp_video_frame: snp.NumpyStorage
        for timestamp, images in iter_frames():
            images = list(images)

            # preprocess
            for i in range(len(images)):
                images[i] = cv2.resize(images[i], None, fx=RESIZE_RATIO, fy=RESIZE_RATIO)
                images[i] = images[i].astype(np.float32)
                images[i] /= 256.0
                assert images[i].max() < 1.0, images.max()

            # dump original
            _, original, _ = images
            original = original.copy()

            # generate diff
            x, y, z = images
            diff_img = np.sqrt(np.sqrt(np.square(y - x) * np.square(z - y)))
            diff_img_gaussian = cv2.GaussianBlur(diff_img, (3, 3), 1)
            motion = np.clip(diff_img_gaussian * 5, 0.0, 1.0 - 1e-6)

            # dump
            snp_video_frame[i] = snp_context.VideoFrameNumpyStorageEntry(
                original=original,
                motion=motion,
                timestamp=timestamp
            )
