import glob
import os
import time

import cv2
import decord
import numpy as np
from tqdm import tqdm

import async_writer


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

    def iter_frames():
        frame_index = 0
        bar = tqdm(np.arange(0, frame_count, STEP))
        while True:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, frame = cap.read()
            if not ret:
                break
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pos = frame_index / frame_rate
            yield dict(
                image=image,
                pos=pos,
                index=frame_index
            )
            frame_index += STEP
            bar.update(1)
            bar.set_description(f'pos={int(pos) // 60}:{int(pos) % 60:02}.{int((pos - int(pos)) * 1000):03}')

    def pre_process(frame):
        image = frame['image'][::2, ::2]
        image = cv2.GaussianBlur(image, )
        image =  image / 256.0
        assert image.max() < 1.0, image.max()
        return update_image(frame, image)


    def iter_pairs(it):
        prev = None
        for current in it:
            if prev is not None:
                yield prev, current
            prev = current

    def update_image(src, new_image):
        return dict(
            image=new_image,
            pos=src['pos'],
            index=src['index']
        )

    def scale(n):
        def f(frame):
            image = frame['image'] * n
            image = np.clip(image, 0.0, 1.0 - 1e-6)
            return update_image(frame, image)

        return f

    def diff_two_frames(args):
        prev, current = args
        diff = np.square(current['image'] - prev['image'])
        return update_image(current, diff)

    def prod_two_frames(args):
        prev, current = args
        prod = np.sqrt(prev['image'] * current['image'])
        return update_image(current, prod)

    def head(args):
        return args[1]

    it = iter_frames()
    it = map(pre_process, it)
    it = iter_pairs(it)
    it = map(diff_two_frames, it)
    it = iter_pairs(it)
    it = map(prod_two_frames, it)
    it = map(scale(5000.0), it)

    with async_writer.AsyncVideoFrameWriter(
            './main_diff_generator_bug_fix_out.mp4',
            fps=frame_rate / STEP
    ) as writer:
        tot_v = []
        tot_h = []
        tss = []

        DST_PATH = f'out/{video_name}_{output_timestamp}.npz'

        def dump():
            np.savez(DST_PATH, tot_v, tot_h, tss)

        for i, frame in enumerate(it):
            img = frame['image']
            tot_v.append(img.mean(axis=2).mean(axis=0))
            tot_h.append(img.mean(axis=2).mean(axis=1))
            tss.append(frame['pos'])
            if (i + 1) % 10 == 0:
                dump()

            a = frame['image'] * 256.0
            assert np.all(a < 256.0)
            writer.write(a.astype(np.uint8))

        dump()


# # 各フレームをパイプライン処理
# tot = []
# tss = []
#
# for i, product in enumerate(it):
#     tot.append(product['result'].mean(axis=2).mean(axis=0))
#     tss.append(product.position)
#     if (i + 1) % 30 == 0:
#         np.savez(DST_PATH, np.vstack(tot), np.array(tss))
# np.savez(DST_PATH, np.vstack(tot), np.array(tss))


if __name__ == '__main__':
    main()
