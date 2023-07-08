# デスクトップの./idsttvideos/singlesの中にある動画から一つを選び
# その先頭1000フレームまで`out.mp4`に毎`STEP`フレームおきに書き出すサンプル

import glob
import itertools
import os
import time
import numpy as np

import imageio as iio
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm

from extract import VideoFrameReader

import frame_processor as fp

from typing import Iterable, Union, Optional, TypeVar

# 動画を見つける
path = os.path.expanduser(r'~\Desktop/idsttvideos/singles')
glob_pattern = os.path.join(path, r'**/*.mp4')
video_path = glob.glob(glob_pattern, recursive=True)[0]

# video_pathに動画のパスが入る
print(video_path)

# 動画のパスからVideoFrameReaderインスタンスを生成
vfr = VideoFrameReader(video_path)


@fp.unary_mapping_stage
def gaussian_filter(frame):
    return cv2.GaussianBlur(frame, (51, 51), 100)


@fp.binary_mapping_stage
def square_diff_two_frames(prev_frame, cur_frame):
    return np.square(cur_frame - prev_frame)


@fp.binary_mapping_stage
def prod_two_frames(prev_frame, cur_frame):
    return cur_frame * prev_frame


@fp.unary_mapping_stage
def to_bgr2rgb(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def clamp(v_max):
    @fp.unary_mapping_stage
    def stage(frame):
        return np.clip(frame / v_max, 0, 1.0 - 1.0 / 256.0)

    return stage


# パイプライン生成
process = fp.FramePipeline([
    # RGBへ
    fp.unary_mapper(to_bgr2rgb),
    # プロデューサ：source
    fp.FramePipeline.produce('source'),
    # [0, 256)を[0.0, 1.0)に変換
    fp.to_float,
    # ガウスフィルタに通す
    fp.unary_mapper(gaussian_filter),
    # 前後のフレームをペアに
    fp.pair_generator,
    # ペアの差をとって差分フレームへ
    fp.binary_mapper(square_diff_two_frames),
    # 前後の差分フレームをペアに
    fp.pair_generator,
    # ペアの積をとる
    fp.binary_mapper(prod_two_frames),
    # クランプ処理
    fp.unary_mapper(clamp(0.001)),
    # [0.0, 1.0)を[0, 256)に変換
    fp.to_uint,
    # プロデューサ：result
    fp.FramePipeline.produce('result'),
])


def main():
    # フレームのパイプラインを生成
    START, STOP, STEP = 0, 600, 2
    it = process(vfr[START:STOP:STEP])

    # 書き出し先の作成
    out = iio.get_writer(
        'out.mp4',
        format='FFMPEG',
        fps=vfr.fps / STEP,
        mode='I',
        codec='h264',
    )

    # 各フレームをパイプライン処理
    t = []
    tot = []
    F_STEP = 4
    for i, products in enumerate(tqdm(it, total=(STOP - START) // STEP)):
        st, sf = products['source']
        pt, pf = products['result']

        # パイプラインを通ってきた各フレームに対して処理
        # print(i, pt, st, pf.shape, sf.shape)

        t.append(st)

        tot.append(pf.mean(axis=2).mean(axis=0))

        h = sf[::F_STEP, ::F_STEP].shape[0]
        tot_vs = np.clip(np.vstack(tot) * 3, 0, 255)[::1, ::10]
        graph = np.zeros((h, tot_vs.shape[1], 3), dtype=np.uint8)
        tot_vs = tot_vs[-h:, :]
        graph[:tot_vs.shape[0], :tot_vs.shape[1], :] = np.dstack(
            [tot_vs[:tot_vs.shape[0], :tot_vs.shape[1]]] * 3)
        graph[:, :, [0, 2]] = 0
        graph[tot_vs.shape[0] - 1, :, 0] = 255
        out.append_data(
            cv2.cvtColor(
                np.concatenate(
                    [
                        sf[::F_STEP, ::F_STEP],
                        pf[::F_STEP, ::F_STEP],
                        graph
                    ],
                    axis=1
                ),
                cv2.COLOR_BGR2RGB)
        )

        if i % 30 == 0:
            np.save('out.npy', np.vstack(tot))
    np.save('out.npy', np.vstack(tot))

    # 書き出し先を閉じる
    out.close()


if __name__ == '__main__':
    main()
