import contextlib

import matplotlib.pyplot as plt
import numpy as np

np.set_printoptions(suppress=True)

import train_input

import dataset
from functools import cached_property
from tqdm import tqdm

np.set_printoptions(suppress=True)


def process_rect(rect):
    w = rect[1].stop - rect[1].start
    aw = int(w * 1.0)
    return slice(rect[0].start, rect[0].stop), slice(rect[1].start - aw, rect[1].stop + aw)


class FeatureGenerator:
    def __init__(self, video_name):
        self.__video_name = video_name
        self.__cache = dataset.PickleCache(video_name)

    @contextlib.contextmanager
    def __frame_store(self) -> dataset.VideoFrameStorage:
        with dataset.VideoFrameStorage(
                dataset.get_video_frame_dump_dir_path(video_name=self.__video_name),
                mode='r',
        ) as vf_store:
            yield vf_store

    @contextlib.contextmanager
    def __motion_store(self) -> dataset.MotionStorage:
        with dataset.MotionStorage(
                dataset.get_motion_dump_dir_path(video_name=self.__video_name),
                mode='r',
        ) as m_store:
            yield m_store

    @cached_property
    def __timestamp(self):
        with self.__motion_store() as s:
            return s.get_all_of('timestamp')

    def __iter_diff_frames(self):
        rect = process_rect(train_input.load_rect(self.__video_name))

        with self.__frame_store() as s:
            timestamp = s.get_all_of('timestamp')[:-1]
            assert timestamp.shape == self.__timestamp.shape
            assert np.all(timestamp == self.__timestamp)
            for i in range(s.count() - 1):
                yield s.get(i)['motion'][rect]

    def __iter_motion_vector(self):
        with self.__motion_store() as s:
            start = s.get_all_of('start')
            end = s.get_all_of('end')
            for s, e in zip(start, end):
                yield s, e

    N_DIFF_FRAME_SPLIT = 5

    @classmethod
    def __iter_split_slices(cls, shape):
        w, h, *_ = shape
        nw = w // cls.N_DIFF_FRAME_SPLIT
        nh = h // cls.N_DIFF_FRAME_SPLIT
        for i in range(cls.N_DIFF_FRAME_SPLIT):
            si = slice(nw * i, None if i == cls.N_DIFF_FRAME_SPLIT - 1 else nw * (i + 1))
            for j in range(cls.N_DIFF_FRAME_SPLIT):
                sj = slice(nh * j, None if j == cls.N_DIFF_FRAME_SPLIT - 1 else nh * (j + 1))
                yield si, sj

    @classmethod
    def __array_split_mean_2d(cls, a):
        return np.array([a[s].mean() for s in cls.__iter_split_slices(a.shape)])

    N_DIFF_FRAME_FEATURE = 3

    def iter_diff_frame_feature(self):
        for diff in self.__iter_diff_frames():
            split_mean = self.__array_split_mean_2d(diff)
            split_mean_sorted = np.sort(split_mean)
            split_mean_arg_sorted = np.argsort(split_mean)
            diff_frame_feature = np.concatenate([
                split_mean_sorted[-self.N_DIFF_FRAME_FEATURE:],
                (split_mean_arg_sorted[-self.N_DIFF_FRAME_FEATURE:] % self.N_DIFF_FRAME_SPLIT),
                (split_mean_arg_sorted[-self.N_DIFF_FRAME_FEATURE:] // self.N_DIFF_FRAME_SPLIT)
            ])
            yield diff_frame_feature

    N_MOTION_SPLIT = 5
    N_MOTION_FEATURE_BLOCK_SIZE = 3
    assert N_MOTION_FEATURE_BLOCK_SIZE % 2 == 1 and N_MOTION_FEATURE_BLOCK_SIZE >= 3
    N_MOTION_FEATURE_BLOCKS = 11
    MOTION_FEATURE_WIDTH = N_MOTION_FEATURE_BLOCK_SIZE * N_MOTION_FEATURE_BLOCKS
    assert MOTION_FEATURE_WIDTH % 2 == 1

    def iter_motion_vector_feature(self):
        rect = process_rect(train_input.load_rect(self.__video_name))

        def split_vertically(n_split, offset, height, points):
            axis = 0
            n = height // n_split
            criteria = points[:, axis] - offset
            nth_split = criteria.astype(int) // n
            return tuple(
                np.where(np.minimum(nth_split, n_split - 1) == i)[0]
                for i in range(n_split)
            )

        vals = []
        for start, end in self.__iter_motion_vector():
            start_split_arg = split_vertically(
                n_split=self.N_MOTION_SPLIT,
                offset=rect[0].start,
                height=rect[0].stop - rect[0].start,
                points=start
            )

            val_split = []
            for j, arg in enumerate(start_split_arg):
                s, e = start[arg], end[arg]
                vec = e - s
                val_split.append(np.nanmean(vec[:, 0]))
            vals.append(val_split)

        vals = np.array(vals).T
        vals[np.isnan(vals)] = 0

        vals_slide = np.lib.stride_tricks.sliding_window_view(
            np.pad(
                vals,
                pad_width=(
                    (0, 0),
                    (self.MOTION_FEATURE_WIDTH // 2, self.MOTION_FEATURE_WIDTH // 2)
                ),
                mode='reflect'
            ),
            window_shape=(vals.shape[0], self.MOTION_FEATURE_WIDTH)
        )[0]
        vals_slide = np.swapaxes(vals_slide, 0, 1)
        print(f'{vals_slide.shape=}')
        block_split = np.stack(np.split(vals_slide, self.N_MOTION_FEATURE_BLOCKS, axis=2), axis=3)
        print(f'{block_split.shape=}')
        mean = block_split.mean(axis=2)

        # names = [
        #     f'diff {i}-{j}'
        #     for i in range(self.N_MOTION_FEATURE_BLOCKS)
        #     for j in range(self.N_MOTION_FEATURE_BLOCKS)
        #     if i > j
        # ]
        # print(names)

        def block_diff_feature(blocks):
            assert blocks.ndim == 1, blocks.shape
            n = blocks.size
            lst = [blocks[i] - blocks[j] for i in range(n) for j in range(n) if i > j]

            return np.array(lst)

        diff_features = np.apply_along_axis(block_diff_feature, 2, mean)
        features = np.concatenate([mean, diff_features], axis=2)
        features = np.concatenate(features, axis=1)
        print(f'{features.shape=}')

        for feature in features:
            yield feature

    def create_feature_vector(self):
        def iter_feature_vector():
            feature_iters = [
                self.iter_diff_frame_feature(),
                self.iter_motion_vector_feature()
            ]

            for fs in tqdm(zip(*feature_iters)):
                feature = np.concatenate(fs)
                yield feature

        return np.stack(list(iter_feature_vector()))

    def create(self, label=False):
        data = self.__cache.load('feature')
        if data is None:
            data = dict(
                timestamp=self.__timestamp,
                feature=self.create_feature_vector()
            )
            self.__cache.dump('feature', data)
        data['label'] = train_input.load_rally_mask(
            f'./train/iDSTTVideoAnalysis_{self.__video_name}.csv',
            timestamps=self.__timestamp
        )[1].astype(bool) if label else None
        return data


Y_DATA_MARGIN = 10


def create_y_data(rally_mask):
    rm = rally_mask.astype(int)
    mask = (rm[1:] - rm[:-1]) > 0
    index_rally_begin = np.where(mask)[0]

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


if __name__ == '__main__':
    video_names = [
        '20230205_04_Narumoto_Harimoto',
        '20230219_03_Narumoto_Ito',
        '20230225_02_Matsushima_Ando'
    ]

    a = FeatureGenerator('20230205_04_Narumoto_Harimoto').create(label=True)
    x, y = a['feature'], create_y_data(a['label']).astype(bool)
    y_src = y.copy()
    idx = np.arange(y.size)
    sample_idx = np.concatenate([
        np.random.choice(idx[~y], y.sum(), replace=True),
        idx[y]
    ])
    x, y = x[sample_idx], y[sample_idx]
    print(y.sum(), (~y).sum())

    from sklearn.ensemble import RandomForestClassifier

    cl = RandomForestClassifier(
        max_depth=3,
        n_estimators=256,
        verbose=True,
        random_state=42,
        n_jobs=4
    )

    cl.fit(x, y)

    vn_test = '20230219_03_Narumoto_Ito'
    a = FeatureGenerator(vn_test).create(label=False)
    x = a['feature']

    y_pred = cl.predict(x)

    fig, axes = plt.subplots(2, 1, figsize=(100, 4), sharex=True)
    axes[0].imshow(np.tile(y_pred[:, None], 30).T)
    axes[1].imshow(np.tile(y_src[:, None], 30).T)
    plt.tight_layout()
    plt.show()

    fps = 1 / np.diff(a['timestamp']).mean()
    print(f'{fps=}')

    from PIL import Image

    with dataset.VideoFrameStorage(
            dataset.get_video_frame_dump_dir_path(vn_test),
            mode='r'
    ) as vf_store:
        images = []

        STEP = 3
        FAST = 2
        for i in tqdm(range(0, int(a['timestamp'].size * 0.3), STEP)):
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
