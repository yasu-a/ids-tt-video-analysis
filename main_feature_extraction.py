import sys

import matplotlib.pyplot as plt
import numpy as np

np.set_printoptions(suppress=True)

import train_input

import dataset

from tqdm import tqdm

with dataset.VideoFrameStorage(
        dataset.get_video_frame_dump_dir_path(),
        mode='r'
) as vf_storage:
    timestamp = vf_storage.get_all_of('timestamp')

train_input_df, rally_mask = train_input.load_rally_mask(
    './train/iDSTTVideoAnalysis_20230205_04_Narumoto_Harimoto.csv',
    timestamp
)


def split_vertically(n_split, offset, height, points):
    axis = 0
    n = height // n_split
    criteria = points[:, axis] - offset
    nth_split = criteria.astype(int) // n
    return tuple(
        np.where(np.minimum(nth_split, n_split - 1) == i)[0]
        for i in range(n_split)
    )


def create_rect():
    r = slice(70, 260), slice(180, 255)  # height, width

    # height: 奥の選手の頭から手前の選手の足がすっぽり入るように
    # width: ネットの部分の卓球台の幅に合うように

    def process_rect(rect):
        w = rect[1].stop - rect[1].start
        aw = int(w * 1.0)
        return slice(rect[0].start, rect[0].stop), slice(rect[1].start - aw, rect[1].stop + aw)

    r = process_rect(r)

    return r


rect = create_rect()

with dataset.MotionStorage(
        dataset.get_motion_dump_dir_path(),
        mode='r',
) as m_store:
    N_SPLIT = 5
    N_MAX_MOTION = 32
    mv_x = [[] for _ in range(N_SPLIT)]
    fs, fe = 0, 3000
    for i in tqdm(range(fs, fe)):
        data_dct = m_store.get(i)
        start, end = data_dct['start'], data_dct['end']
        start, end = start[~np.isnan(start[:, 0])], end[~np.isnan(end[:, 0])]
        assert len(start) == len(end), (start.shape, end.shape)

        start_split_arg = split_vertically(
            n_split=N_SPLIT,
            offset=rect[0].start,
            height=rect[0].stop - rect[0].start,
            points=start
        )

        for j, arg in enumerate(start_split_arg):
            s, e = start[arg], end[arg]
            x, y, ex, ey = s[:, 0], s[:, 1], e[:, 0], e[:, 1]
            dx = ex - x
            assert len(dx) < N_MAX_MOTION, len(dx)
            pad = np.full(N_MAX_MOTION - len(dx), np.nan)
            dx = np.concatenate([dx, pad])
            if np.all(np.isnan(dx)):
                dx[0] = 0
            mv_x[j].append(list(dx))

    mv_x = np.array(mv_x)

    print(mv_x.shape)
    features = dict(
        mean=np.nanmean(mv_x, axis=2),
        std=np.nanstd(mv_x, axis=2)
    )
    print(features['mean'].shape)

    PLOT_FIGURE = False
    if PLOT_FIGURE:
        def plot_figure():
            fig, axes = plt.subplots(N_SPLIT + 1, 1, figsize=(200, 10), sharex=True)
            axes[0].set_xlim(0, fe - fs)
            axes[0].imshow(np.tile(rally_mask[fs:fe][:, None], 30).T)
            for i in range(N_SPLIT):
                axes[i + 1].plot(features['mean'][i], label='mean', alpha=0.7)
                axes[i + 1].plot(features['std'][i], label='std', alpha=0.7)
                axes[i + 1].legend()

            def ax_edit_ticklabel(ax):
                ax.set_xticklabels(timestamp[fs:][ax.get_xticks().astype(int)].round(1),
                                   rotation=90)

            ax_edit_ticklabel(axes[0])

            fig.tight_layout()
            fig.show()


        plot_figure()

    print({k: v.shape for k, v in features.items()})

    WINDOW_WIDTH_HALF = 6
    WINDOW_WIDTH_FULL = WINDOW_WIDTH_HALF + 1 + WINDOW_WIDTH_HALF


    def extract_motion_energy(src):
        assert src.ndim == 2, src.shape

        src_single_pad = np.pad(
            src,
            pad_width=((0, 0), (1, 1)),
            mode='edge'
        )

        ts = np.pad(
            timestamp[fs:fe],
            pad_width=((1, 1),),
            mode='reflect'
        )
        ts_diff = np.abs(ts[2:] - ts[1:-1]) + np.abs(ts[1:-1] - ts[:-2])
        diff = (src_single_pad[:, 2:] - src_single_pad[:, :-2]) / ts_diff

        slide = np.lib.stride_tricks.sliding_window_view(
            np.pad(
                diff,
                ((0, 0), (WINDOW_WIDTH_HALF, WINDOW_WIDTH_HALF)),
                mode='reflect'
            ),
            window_shape=(src.shape[0], WINDOW_WIDTH_FULL),
        )[0]

        motion_energy = np.square(slide)

        i = 1
        fig, axes = plt.subplots(7, 1, figsize=(100, 25), sharex=True)
        axes[0].imshow(np.tile(rally_mask[fs:fe][:, None], 30).T)
        axes[-1].imshow(np.tile(rally_mask[fs:fe][:, None], 30).T)
        axes[1].plot(src[i])
        axes[2].plot(motion_energy[:, i, WINDOW_WIDTH_HALF])
        axes[3].plot(motion_energy.max(axis=2)[:, i])
        axes[4].plot(motion_energy.mean(axis=2)[:, i])
        axes[5].plot(motion_energy.min(axis=2)[:, i])
        fig.tight_layout()
        fig.show()


    def extract_motion_past_and_future(src):
        assert src.ndim == 2, src.shape

        src_single_pad = np.pad(
            src,
            pad_width=((0, 0), (1, 1)),
            mode='edge'
        )

        ts = np.pad(
            timestamp[fs:fe],
            pad_width=((1, 1),),
            mode='reflect'
        )
        ts_diff = np.abs(ts[2:] - ts[1:-1]) + np.abs(ts[1:-1] - ts[:-2])
        diff = (src_single_pad[:, 2:] - src_single_pad[:, :-2]) / ts_diff

        slide_past = np.lib.stride_tricks.sliding_window_view(
            np.pad(
                diff,
                ((0, 0), (WINDOW_WIDTH_HALF, 0)),
                mode='reflect'
            ),
            window_shape=(src.shape[0], WINDOW_WIDTH_HALF + 1),
        )[0]
        slide_future = np.lib.stride_tricks.sliding_window_view(
            np.pad(
                diff,
                ((0, 0), (0, WINDOW_WIDTH_HALF)),
                mode='reflect'
            ),
            window_shape=(src.shape[0], WINDOW_WIDTH_HALF + 1),
        )[0]

        eng_past = np.square(slide_past)
        eng_future = np.square(slide_future)

        i = 1
        fig, axes = plt.subplots(5, 1, figsize=(100, 20), sharex=True)
        axes[0].imshow(np.tile(rally_mask[fs:fe][:, None], 30).T)
        axes[1].plot(src[i])
        axes[2].plot(eng_past.max(axis=2)[:, i])
        axes[3].plot(eng_future.max(axis=2)[:, i])
        axes[4].imshow(np.tile(rally_mask[fs:fe][:, None], 30).T)
        fig.tight_layout()
        fig.show()


    def extract_motion_outlier_count(src):
        assert src.ndim == 2, src.shape

        src_pad = np.pad(
            src,
            pad_width=((0, 0), (WINDOW_WIDTH_HALF, WINDOW_WIDTH_HALF))
        )
        slide = np.lib.stride_tricks.sliding_window_view(
            src_pad,
            window_shape=(src.shape[0], WINDOW_WIDTH_FULL)
        )[0]

        print(slide.shape)
        slide = np.swapaxes(slide, 0, 1)
        print(slide.shape)

        mean = slide.mean(axis=2)
        std = slide.std(axis=2)
        top, bottom = mean + std * 2, mean - std * 2

        src_single_pad = np.pad(
            src,
            pad_width=((0, 0), (1, 1))
        )
        cross_top = (
                            (src_single_pad[:, :-2] - top) *
                            (src_single_pad[:, 2:] - top)
                    ) < 0
        cross_bottom = (
                               (src_single_pad[:, :-2] - bottom) *
                               (src_single_pad[:, 2:] - bottom)
                       ) < 0
        cross = cross_top | cross_bottom

        cross_pad = np.pad(
            cross,
            pad_width=((0, 0), (WINDOW_WIDTH_HALF, WINDOW_WIDTH_HALF))
        )
        cross_slide = np.lib.stride_tricks.sliding_window_view(
            cross_pad,
            window_shape=(cross_pad.shape[0], WINDOW_WIDTH_FULL)
        )[0]
        cross_slide = np.swapaxes(cross_slide, 0, 1)

        cross_count = cross_slide.mean(axis=2)

        i = 2
        fig, axes = plt.subplots(5, 1, figsize=(100, 25), sharex=True)
        axes[0].imshow(np.tile(rally_mask[fs:fe][:, None], 30).T)
        axes[1].plot(src[i])
        axes[1].plot(top[i, :])
        axes[1].plot(bottom[i, :])
        axes[2].plot(cross[i, :])
        axes[3].plot(cross_count[i, :])
        fig.tight_layout()
        fig.show()


    extract_motion_energy(features['mean'])
