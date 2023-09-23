import datetime
import glob
import itertools
import json
import os
import time

import cv2
import decord
import numpy as np
from tqdm import tqdm

import async_writer

MEMMAP_PATH = 'G:\iDSTTVideoFrameDump'


def main():
    # video_pathに動画のパスを設定
    video_path = os.path.expanduser(
        r'~/Desktop/idsttvideos/singles\20230205_04_Narumoto_Harimoto.mp4'
    )
    video_name = os.path.splitext(os.path.split(video_path)[1])[0]
    output_timestamp = int(time.time())
    print(video_path)

    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
    print(frame_count, frame_rate)
    STEP = 5

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

        def flag_it():
            pattern = np.zeros(STEP, dtype=int)
            pattern[:3] = [1, 2, 3]

            for index in itertools.count():
                flag = pattern[index % STEP]
                yield index, flag

        bar = tqdm(flag_it(), total=frame_count)
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

    def remove_side(frame):
        return update_image(frame, prev_image=None, next_image=None)

    def diff_frames(frame):
        x, y, z = frame['prev_image'], frame['image'], frame['next_image']
        image = np.sqrt(np.sqrt(np.square(y - x) * np.square(z - y)))
        return update_image(frame, image=image)

    def time_filter(f):
        def wrapper(it):
            for frame in it:
                ts = frame['timestamp']
                if f(ts):
                    yield frame

        return wrapper

    it = iter_frames()  # time_start=21, time_end=105
    it = map(pre_process, it)
    it = map(diff_frames, it)
    it = map(remove_side, it)
    it = map(blur, it)
    it = map(scale(5.0), it)

    with async_writer.AsyncVideoFrameWriter(
            './main_diff_generator_bug_fix_out.mp4',
            fps=frame_rate / STEP
    ) as writer:
        # tot_v = []
        # tot_h = []
        tss = []
        frames = []

        DST_PATH = f'out/{video_name}_{output_timestamp}.npz'

        def dump():
            if len(tss) == 0:
                return

            # np.savez(DST_PATH, np.stack(tot_v), np.stack(tot_h), np.array(tss))
            frames_map = np.memmap(
                os.path.join(MEMMAP_PATH, 'frames.map'),
                mode='w+',
                dtype=np.uint8,
                shape=(len(frames), *frames[0].shape)
            )
            frames_map[:, :, :, :] = frames

            tss_map = np.memmap(
                os.path.join(MEMMAP_PATH, 'tss.map'),
                mode='w+',
                dtype=np.float32,
                shape=(len(tss),)
            )
            tss_map[:] = tss

            with open(os.path.join(MEMMAP_PATH, 'shape.json'), 'w') as f:
                json.dump({
                    'frames': frames_map.shape,
                    'tss': tss_map.shape
                }, f, indent=2)

        for i, frame in enumerate(it):
            img = frame['image']
            # tot_v.append(img.mean(axis=2).mean(axis=0))
            # tot_h.append(img.mean(axis=2).mean(axis=1))
            tss.append(frame['timestamp'])
            frames.append(np.clip((img * 256.0).astype(int), 0, 255).astype(np.uint8))
            if (i + 1) % 128 == 0:
                dump()

            writer.write(frames[-1])

        dump()


if __name__ == '__main__':
    main()
