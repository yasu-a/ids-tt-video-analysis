import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import train_input

np.set_printoptions(suppress=True)


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


if __name__ == '__main__':
    rect = create_rect()

    with dataset.MotionStorage(
            dataset.get_motion_dump_dir_path(),
            mode='r',
    ) as m_store:
        N_SPLIT = 5
        N_MAX_MOTION = 32
        mv_x = [[] for _ in range(N_SPLIT)]
        fs, fe = 0, 13321
        fe = min(fe, m_store.count())
        timestamp = []
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

            timestamp.append(data_dct['timestamp'])

        mv_x = np.array(mv_x)

        print(mv_x.shape)
        features = dict(
            mean=np.nanmean(mv_x, axis=2),
            std=np.nanstd(mv_x, axis=2)
        )
        print(features['mean'].shape)

        timestamp = np.array(timestamp)

        _, rally_mask = train_input.load_rally_mask(
            './train/iDSTTVideoAnalysis_20230205_04_Narumoto_Harimoto.csv',
            timestamp
        )

        del fs
        del fe

        PLOT_FIGURE = False
        if PLOT_FIGURE:
            def plot_figure():
                fig, axes = plt.subplots(N_SPLIT + 1, 1, figsize=(200, 10), sharex='all')
                axes[0].imshow(np.tile(rally_mask[:, None], 30).T)
                for i in range(N_SPLIT):
                    axes[i + 1].plot(features['mean'][i], label='mean', alpha=0.7)
                    axes[i + 1].plot(features['std'][i], label='std', alpha=0.7)
                    axes[i + 1].legend()

                def ax_edit_ticklabel(ax):
                    ax.set_xlim(0, timestamp.size - 1)
                    ts_max = (timestamp.size - 1) // 100 * 100
                    ax.set_xticks(np.linspace(0, ts_max, ts_max // 100))
                    ax.set_xticklabels(timestamp[ax.get_xticks().astype(int)].round(1))
                    ax.tick_params(axis="x", labelrotation=90)

                ax_edit_ticklabel(axes[0])

                fig.tight_layout()
                fig.show()


            plot_figure()

        print({k: v.shape for k, v in features.items()})


        def extract_motion_energy(src):
            assert src.ndim == 2, src.shape

            src_single_pad = np.pad(
                src,
                pad_width=((0, 0), (1, 1)),
                mode='edge'
            )

            ts = np.pad(
                timestamp,
                pad_width=((1, 1),),
                mode='reflect'
            )
            ts_diff = np.abs(ts[2:] - ts[1:-1]) + np.abs(ts[1:-1] - ts[:-2])
            diff = (src_single_pad[:, 2:] - src_single_pad[:, :-2]) / ts_diff

            motion_energy = np.square(diff)
            return motion_energy


        FEATURE_BLOCK_SIZE = 3
        assert FEATURE_BLOCK_SIZE % 2 == 1 and FEATURE_BLOCK_SIZE >= 3
        N_FEATURE_BLOCKS = 11
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

        names = [f'diff {i}-{j}' for i in range(N_FEATURE_BLOCKS) for j in range(N_FEATURE_BLOCKS)
                 if
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

        rm = rally_mask.astype(int)
        mask = (rm[1:] - rm[:-1]) > 0
        index_rally_begin = np.where(mask)[0]

        # fig, axes = plt.subplots(2, 1, figsize=(100, 8))
        # axes[0].plot(mask)
        # axes[1].plot(rm[:-1])
        # fig.show()

        Y_DATA_MARGIN = 10


        def create_y_data() -> np.ndarray:
            assert rm.ndim == 1, rm.shape
            index_delta = np.arange(rm.size)[:, None] - index_rally_begin
            nearest_rally_begin_index = index_rally_begin[np.abs(index_delta).argmin(axis=1)]
            nearest_rally_begin_index_delta = nearest_rally_begin_index - np.arange(rm.size)
            # y = (-Y_DATA_MARGIN < nearest_rally_begin_index_delta) & (
            #             nearest_rally_begin_index_delta < 0)
            y = (0 < nearest_rally_begin_index_delta) & (
                    nearest_rally_begin_index_delta < Y_DATA_MARGIN)
            # y = np.abs(nearest_rally_begin_index_delta) < Y_DATA_MARGIN // 2

            # fig, axes = plt.subplots(3, 1, figsize=(100, 8))
            # axes[0].plot(rm[:-1])
            # axes[1].plot(nearest_rally_begin_index_delta)
            # axes[2].plot(y)
            # fig.show()

            return y


        x = features
        y = create_y_data()

        n_neg, n_pos = (~y).sum(), y.sum()
        print(f'first {x.shape=} {y.shape=} {n_neg=}, {n_pos=}')

        neg_idx, pos_idx = np.where(~y)[0], np.where(y)[0]
        neg_idx = np.random.choice(neg_idx, n_pos, replace=False)
        sample_idx = np.sort(np.concatenate([neg_idx, pos_idx]))
        x = x[sample_idx]
        y = y[sample_idx]

        n_neg, n_pos = (~y).sum(), y.sum()
        print(f'last {x.shape=} {y.shape=} {n_neg=}, {n_pos=}')

        # import sklearn.model_selection
        #
        # x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(
        #     x, y, test_size=0.1, random_state=42
        # )

        SPLIT_RATIO = 0.33
        SIZE = int(y.size * SPLIT_RATIO)
        print(f'{SIZE=}')
        x_train, x_test, y_train, y_test = x[SIZE:], x[:SIZE], y[SIZE:], y[:SIZE]

        from sklearn.ensemble import RandomForestClassifier

        cl = RandomForestClassifier(
            max_depth=3,
            n_estimators=256,
            verbose=True,
            random_state=42,
            n_jobs=4
        )
        cl.fit(x_train, y_train)

        import sklearn.metrics

        y_true = y_test
        y_pred = cl.predict(x_test)
        print(sklearn.metrics.confusion_matrix(y_true, y_pred))
        print(sklearn.metrics.f1_score(y_true, y_pred))

        y_pred = cl.predict(features).astype(int)
        print(y_pred)

        print(rally_mask.shape, y_pred.shape)
        fig, axes = plt.subplots(2, 1, figsize=(100, 3), sharex='all')
        axes[0].imshow(np.tile(rally_mask[:, None], 30).T)
        axes[1].imshow(np.tile(y_pred[:, None], 30).T)


        def ax_edit_ticklabel(ax):
            ax.set_xlim(0, timestamp.size - 1)
            ts_max = (timestamp.size - 1) // 100 * 100
            ax.set_xticks(np.linspace(0, ts_max, ts_max // 100))
            ax.set_xticklabels(timestamp[ax.get_xticks().astype(int)].round(1))
            ax.tick_params(axis="x", labelrotation=90)


        ax_edit_ticklabel(axes[0])

        fig.tight_layout()
        fig.show()

        fps = 1 / np.diff(timestamp).mean()
        print(f'{fps=}')

        from PIL import Image

        with dataset.VideoBaseFrameStorage(
                dataset.get_video_frame_dump_dir_path(),
                mode='r'
        ) as vf_store:
            images = []

            STEP = 3
            FAST = 2
            for i in tqdm(range(0, int(timestamp.size * SPLIT_RATIO), STEP)):
                img = vf_store.get(i)['original']
                if y_pred[i]:
                    img[:, :30:, :] = [0, 255, 0]
                else:
                    img[:, :30:, :] = [0, 0, 255]
                images.append(Image.fromarray(img))

            images[0].save('output.gif',
                           save_all=True, append_images=images[1:], optimize=False,
                           duration=1 / fps * STEP / FAST,
                           loop=0)
