import glob
import os
import time

import cv2
import numpy as np
from tqdm import tqdm

import frame_processor as fp
from async_writer import AsyncVideoFrameWriter
from extract import VideoFrameReader


# FIXME: THIS SCRIPT MAY HAVE A MEMORY LEAK


@fp.stage.each
@fp.mapper.process(fp.IMAGE)
@fp.mapper.unary
def stage_sample_quarter(frame):
    return frame[::2, ::2]


@fp.stage.each
@fp.mapper.process(fp.IMAGE)
@fp.mapper.unary
def stage_gaussian_filter(frame):
    return cv2.GaussianBlur(frame, (31, 31), min(frame.shape) / 10)


@fp.stage.contiguous
@fp.mapper.process(fp.IMAGE)
@fp.mapper.past
def stage_square_diff_two_frames(prev_frame, cur_frame):
    return np.square(cur_frame - prev_frame)


@fp.stage.contiguous
@fp.mapper.process(fp.IMAGE)
@fp.mapper.past
def stage_prod_two_frames(prev_frame, cur_frame):
    return cur_frame * prev_frame


@fp.stage.each
@fp.mapper.process(fp.IMAGE)
@fp.mapper.unary
def stage_bgr2rgb(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


@fp.stage.each
@fp.mapper.process(fp.IMAGE)
@fp.mapper.unary
def stage_rgb2bgr(frame):
    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)


def stage_clamp(v_max):
    @fp.stage.each
    @fp.mapper.process(fp.IMAGE)
    @fp.mapper.unary
    def stage(frame):
        return np.clip(frame / v_max, 0, 1.0 - 1.0 / 256.0)

    return stage


def main():
    # video_pathに動画のパスを設定
    video_path = os.path.expanduser(
        r'~/Desktop/idsttvideos/singles\20230205_04_Narumoto_Harimoto.mp4'
    )
    video_name = os.path.splitext(os.path.split(video_path)[1])[0]
    output_timestamp = int(time.time())
    print(video_path)

    # 動画のパスからVideoFrameReaderインスタンスを生成
    vfr = VideoFrameReader(video_path)

    # パイプライン構築
    pipeline = fp.create_pipeline([
        # 1/4サイズへ
        stage_sample_quarter,
        # プロデューサ：source
        fp.stage.store('source'),
        # BGRへ
        stage_rgb2bgr,
        # [0, 256)を[0.0, 1.0)に変換
        fp.stage.to_double,
        # # ガウスフィルタに通す
        # stage_gaussian_filter,
        # ペアの差をとって差分フレームへ
        stage_square_diff_two_frames,
        # ペアの積をとる
        stage_prod_two_frames,
        # クランプ処理
        stage_clamp(0.00002),
        # [0.0, 1.0)を[0, 256)に変換
        fp.stage.to_uint,
        # RGBへ
        stage_bgr2rgb,
        # プロデューサ：result
        fp.stage.store('result'),
    ])

    # フレームのパイプラインを生成
    START, STOP, STEP = 500, 1300, 5
    it = pipeline(vfr)[::STEP]

    # 各フレームをパイプライン処理
    tot = []
    tss = []
    DST_PATH = f'out/{video_name}_{output_timestamp}.npz'
    for i, product in enumerate(it):
        tot.append(product['result'].mean(axis=2).mean(axis=0))
        tss.append(product.position)
        if (i + 1) % 30 == 0:
            np.savez(DST_PATH, np.vstack(tot), np.array(tss))
    np.savez(DST_PATH, np.vstack(tot), np.array(tss))


if __name__ == '__main__':
    main()
