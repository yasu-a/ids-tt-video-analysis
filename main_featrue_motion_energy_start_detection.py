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
    print(m_store.get_all_of('timestamp'))
    N_SPLIT = 5
    N_MAX_MOTION = 32
    mv_x = [[] for _ in range(N_SPLIT)]
    fs, fe = 0, 13000
    fe = min(fe, m_store.count())
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
            dy = ey - y
            dx = dx[~((dx == 0) & (dy == 0))]
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

        motion_energy = np.square(diff)
        return motion_energy

        # slide = np.lib.stride_tricks.sliding_window_view(
        #     np.pad(
        #         diff,
        #         ((0, 0), (WINDOW_WIDTH_HALF, WINDOW_WIDTH_HALF)),
        #         mode='reflect'
        #     ),
        #     window_shape=(src.shape[0], WINDOW_WIDTH_FULL),
        # )[0]
        # slide = np.swapaxes(slide, 0, 1)
        #
        # motion_energy = np.square(slide)

        # i = 1
        # fig, axes = plt.subplots(7, 1, figsize=(100, 25), sharex=True)
        # axes[0].imshow(np.tile(rally_mask[fs:fe][:, None], 30).T)
        # axes[-1].imshow(np.tile(rally_mask[fs:fe][:, None], 30).T)
        # axes[1].plot(src[i])
        # axes[2].plot(motion_energy[:, i, WINDOW_WIDTH_HALF])
        # axes[3].plot(motion_energy.max(axis=2)[:, i])
        # axes[4].plot(motion_energy.mean(axis=2)[:, i])
        # axes[5].plot(motion_energy.min(axis=2)[:, i])
        # fig.tight_layout()
        # fig.show()

        # return motion_energy.mean(axis=2)


    FEATURE_BLOCK_SIZE = 5
    assert FEATURE_BLOCK_SIZE % 2 == 1 and FEATURE_BLOCK_SIZE >= 3
    N_FEATURE_BLOCKS = 5
    FEATURE_WIDTH = FEATURE_BLOCK_SIZE * N_FEATURE_BLOCKS
    assert FEATURE_WIDTH % 2 == 1

    me = extract_motion_energy(features['mean'])
    me_slide = np.lib.stride_tricks.sliding_window_view(
        np.pad(
            me,
            pad_width=((0, 0), (FEATURE_WIDTH // 2, FEATURE_WIDTH // 2)),
            mode='reflect'
        ),
        window_shape=(me.shape[0], FEATURE_WIDTH)
    )[0]
    me_slide = np.swapaxes(me_slide, 0, 1)
    print(f'{me_slide.shape=}')
    block_split = np.stack(np.split(me_slide, N_FEATURE_BLOCKS, axis=2), axis=3)
    print(f'{block_split.shape=}')
    mean = block_split.mean(axis=2)

    names = [f'diff {i}-{j}' for i in range(N_FEATURE_BLOCKS) for j in range(N_FEATURE_BLOCKS) if
             i > j]
    print(names)


    def block_diff_feature(blocks):
        assert blocks.ndim == 1, blocks.shape
        n = blocks.size
        lst = [blocks[i] - blocks[j] for i in range(n) for j in range(n) if i > j]

        return np.array(lst)


    print(mean.shape)
    diff_features = np.apply_along_axis(block_diff_feature, 2, mean)
    print(diff_features.shape)
    features = np.concatenate([diff_features], axis=2)
    print(features.shape)
    features = np.concatenate(features, axis=1)
    print(features.shape)

    rm = rally_mask[fs:fe].astype(int)
    mask = (rm[1:] - rm[:-1]) > 0
    index_rally_begin = np.where(mask)[0]


    # fig, axes = plt.subplots(2, 1, figsize=(100, 8))
    # axes[0].plot(mask)
    # axes[1].plot(rm[:-1])
    # fig.show()

    def create_y_data():
        assert rm.ndim == 1, rm.shape
        index_delta = np.arange(rm.size)[:, None] - index_rally_begin
        nearest_rally_begin_index = index_rally_begin[np.abs(index_delta).argmin(axis=1)]
        nearest_rally_begin_index_delta = np.abs(nearest_rally_begin_index - np.arange(rm.size))
        y = nearest_rally_begin_index_delta < FEATURE_BLOCK_SIZE

        # fig, axes = plt.subplots(3, 1, figsize=(100, 8))
        # axes[0].plot(rm[:-1])
        # axes[1].plot(nearest_rally_begin_index_delta)
        # axes[2].plot(y)
        # fig.show()

        return y


    x = features
    y = create_y_data()

    target_idx = np.random.choice(np.where(~y)[0], size=y.sum() * 8, replace=False)
    mask = np.zeros(y.size, dtype=bool)
    mask[target_idx] = 1
    mask[y] = 1
    x = x[mask]
    y = y[mask]

    # plt.figure()
    # j = 11
    # plt.hist(x[:, j][y], bins=128, alpha=0.5)
    # plt.hist(x[:, j][~y], bins=128, alpha=0.5)
    # plt.show()

    n_neg, n_pos = (~y).sum(), y.sum()
    print(f'first {x.shape=} {y.shape=} {n_neg=}, {n_pos=}')

    import sklearn.model_selection

    x_train_0, x_test_0, y_train_0, y_test_0 = sklearn.model_selection.train_test_split(
        x[~y], y[~y], test_size=0.1, random_state=42
    )

    x_train_1, x_test_1, y_train_1, y_test_1 = sklearn.model_selection.train_test_split(
        x[y], y[y], test_size=0.1, random_state=42
    )

    x_train, x_test, y_train, y_test = np.array([]), np.array([]), np.array([]), np.array([])
    reduce = np.random
    for n in 'x_train, x_test, y_train, y_test'.split(','):
        a = locals()[n.strip() + '_0']
        b = locals()[n.strip() + '_1']
        locals()[n.strip()] = np.concatenate([a, b], axis=0)

    n_neg, n_pos = (~y_train).sum(), y_train.sum()
    print(f'{x_train.shape} {y_train.shape} {n_neg=}, {n_pos=}')
    n_neg, n_pos = (~y_train).sum(), y_train.sum()
    pos_idx = np.where(y_train)[0]
    pos_idx_samples = np.random.choice(pos_idx, size=n_neg - n_pos, replace=True)
    x_train = np.concatenate([x_train, x_train[pos_idx_samples]], axis=0)
    y_train = np.concatenate([y_train, y_train[pos_idx_samples]], axis=0)
    n_neg, n_pos = (~y_train).sum(), y_train.sum()
    print(f'{x_train.shape} {y_train.shape} {n_neg=}, {n_pos=}')

    from sklearn.ensemble import RandomForestClassifier

    cl = RandomForestClassifier(
        max_depth=3,
        n_estimators=128,
        verbose=True,
        random_state=42,
        n_jobs=4
    )
    cl.fit(x_train, y_train)

    import sklearn.metrics

    y_true = y_test
    y_pred = cl.predict(x_test)
    print(sklearn.metrics.confusion_matrix(y_true, y_pred))
