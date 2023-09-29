import datetime
import itertools
import sys
import threading

import cv2
import numpy as np
from tqdm import tqdm

import dataset

SPECIFIED_FPS = 29.97


def iter_results(video_path):
    # create VideoCapture instance
    cap = cv2.VideoCapture(video_path)

    # extract meta
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_rate = float(cap.get(cv2.CAP_PROP_FPS))
    assert abs(frame_rate - SPECIFIED_FPS) < 1e-4, frame_rate

    # process config
    STEP = 5

    # output frame index and characteristic
    def flag_it():
        pattern = np.zeros(STEP, dtype=int)
        pattern[:3] = [1, 2, 3]

        for index in range(frame_count):
            flag = pattern[index % STEP]
            yield index, flag

    index_flag_pairs = list(flag_it())
    n_output = int(sum(f == 3 for idx, f in index_flag_pairs))

    # yield meta
    yield dict(
        frame_count=frame_count,
        frame_rate=frame_rate,
        n_output=n_output
    )

    # utils
    def iter_frames(time_start=None, time_end=None):
        def retrieve_frame():
            _, image = cap.retrieve()
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return image

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        stack = []
        center_index = 0
        center_timestamp = None

        yield_count = 0

        bar = tqdm(index_flag_pairs)
        for index, flag in bar:
            if not cap.grab():
                break

            timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            bar.set_description(f'{datetime.timedelta(seconds=timestamp)!s} {yield_count:5}')

            if time_end is not None and time_end < timestamp:
                break
            if time_start is not None and timestamp < time_start:
                continue

            if flag == 1 and len(stack) == 0:
                stack.append(retrieve_frame())
            elif flag == 2 and len(stack) == 1:
                stack.append(retrieve_frame())
                center_index = index
                center_timestamp = timestamp
            elif flag == 3 and len(stack) == 2:
                stack.append(retrieve_frame())
                yield_count += 1
                yield dict(
                    prev_image=stack[0],
                    image=stack[1],
                    next_image=stack[2],
                    timestamp=center_timestamp,
                    center_index=center_index
                )
                stack.clear()

    def update_image(src, **kwargs):
        d = dict(src)
        d.update(kwargs)
        return d

    def vectorize_image_set(image_mapper):
        def frame_process(frame):
            dct = {}
            for k in ('prev_image', 'image', 'next_image'):
                if frame[k] is None:
                    continue
                dct[k] = image_mapper(frame[k])
            return update_image(frame, **dct)

        return frame_process

    @vectorize_image_set
    def pre_process(image):
        image = cv2.resize(image, None, fx=0.3, fy=0.3)
        image = image / 256.0
        assert image.max() < 1.0, image.max()
        return image

    @vectorize_image_set
    def blur(image):
        image = cv2.GaussianBlur(image, (3, 3), 1)
        return image

    def scale(n):
        @vectorize_image_set
        def f(image):
            return np.clip(image * n, 0.0, 1.0 - 1e-6)

        return f

    def remove_side(fr):
        return update_image(fr, prev_image=None, next_image=None)

    def diff_frames(fr):
        x, y, z = fr['prev_image'], fr['image'], fr['next_image']
        image = np.sqrt(np.sqrt(np.square(y - x) * np.square(z - y)))
        return update_image(fr, image=image)

    def dump_image(dct_key):
        def f(src):
            assert isinstance(src, dict), type(src)
            src[dct_key] = src['image'].copy()
            return src

        return f

    def to_uint8(a):
        return np.clip((a * 256.0).astype(int), 0, 255).astype(np.uint8)

    # generate iterator
    it = iter_frames()
    it = map(pre_process, it)
    it = map(dump_image('original'), it)
    it = map(diff_frames, it)
    it = map(remove_side, it)
    it = map(blur, it)
    it = map(scale(5.0), it)

    # iterate and yield extracted data
    for i, frame in enumerate(it):
        data_dct = dict(
            original=to_uint8(frame['original']),
            motion=to_uint8(frame['image']),
            timestamp=frame['timestamp']
        )
        yield i, data_dct


def process(video_name, signal: list = None):
    frame_dump_io = dataset.FrameDumpIO(video_name=video_name)
    it = iter_results(frame_dump_io.video_path)
    meta = next(it)
    print(meta)
    with frame_dump_io.get_storage(
            mode='w',
            max_entries=meta['n_output']
    ) as storage:
        for i, data_dct in it:
            storage.put(i, data_dct)
            if signal and signal[0]:
                print('PROCESS INTERRUPTED')
                break
        print('PROCESS COMPLETED')


def main(video_name):
    signal = [False]

    th = threading.Thread(
        target=process,
        args=(video_name, signal),
        name='frame-dump-process'
    )
    th.start()

    print('PRESS ENTER TO EXIT')
    input()
    signal[0] = True
    th.join()


if __name__ == '__main__':
    _, *args = sys.argv
    if args:
        video_name = args[0]
    else:
        video_name = None
    main(video_name=video_name)
