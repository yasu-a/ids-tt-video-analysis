import glob
import os

import cv2
import numpy as np
from tqdm import tqdm

import frame_processor as fp
from async_writer import AsyncVideoFrameWriter
from extract import VideoFrameReader


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
    # 動画を見つける
    path = os.path.expanduser(r'~\Desktop/idsttvideos/singles')
    glob_pattern = os.path.join(path, r'**/*.mp4')
    video_path = glob.glob(glob_pattern, recursive=True)[0]

    # video_pathに動画のパスが入る
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
    START, STOP, STEP = 900, 1120, 1
    it = pipeline(vfr)[START:STOP:STEP]

    # 書き出し先の作成
    with AsyncVideoFrameWriter(path='out.mp4', fps=vfr.fps / STEP) as writer:

        # 各フレームをパイプライン処理
        t = []
        tot = []
        F_STEP = 2
        for i, product in enumerate(tqdm(it, total=(STOP - START) // STEP)):
            st = product.position
            sf = product['source']
            pf = product['result']

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

            out_frame = np.concatenate(
                [
                    sf[::F_STEP, ::F_STEP],
                    pf[::F_STEP, ::F_STEP],
                    graph
                ],
                axis=1
            )
            writer.write(out_frame)

            if i % 30 == 0:
                np.save('out.npy', np.vstack(tot))

        if len(tot) > 0:
            np.save('out.npy', np.vstack(tot))


if __name__ == '__main__':
    main()
