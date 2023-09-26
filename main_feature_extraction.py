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
    fs, fe = 100, 6000
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

    fig, axes = plt.subplots(N_SPLIT + 1, 1, figsize=(300, 15), sharex=True)
    axes[0].set_xlim(0, fe - fs)
    axes[0].imshow(np.tile(rally_mask[fs:fe][:, None], 30).T)
    for i in range(N_SPLIT):
        axes[i + 1].plot(features['mean'][i], label='mean', alpha=0.7)
        axes[i + 1].plot(features['std'][i], label='std', alpha=0.7)
        axes[i + 1].legend()


    def ax_edit_ticklabel(ax):
        ax.set_xticklabels(timestamp[fs:][ax.get_xticks().astype(int)].round(1), rotation=90)


    ax_edit_ticklabel(axes[0])

    fig.tight_layout()
    fig.show()
