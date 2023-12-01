import functools
from dataclasses import dataclass
from pprint import pprint
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage
import skimage.feature
from sklearn.metrics.pairwise import cosine_similarity

import npstorage_context as snp_context
import storage
import storage.npstorage as snp
import train_input
from train_input import RectNormalized


@dataclass(frozen=True)
class PMDetectorParameter:
    # 与えられた画像にかける平均値フィルタの大きさ`画像サイズ // mean_filter_size_factor`
    mean_filter_size_factor: int = 32

    # 輝度極大点を求めるときに`skimage.feature.peak_local_max`の引数`min_distance`に
    # 与える値`画像サイズ // local_max_distance_factor`
    local_max_distance_factor: int = 32

    # 輝度極大点を求めるときに極大点の対象とする最小輝度
    motion_local_max_thresh: float = 0.03

    # キーフレームの相互マッチングのときに，cos距離がこの値以下のマッチングが対象となる。
    # cos距離は`1 - np.clip(cos_distance, 0, 1)`で算出され，0に近いほうが距離が近い。
    mutual_match_max_cos_distance: float = 0.3

    # 切り出すキーフレームの大きさで，中心点から±`key_image_size`の範囲が切り出される。
    # 実際に切り出されるキーフレームの大きさは`key_image_size * 2 + 1`
    key_image_size: int = 32

    # これ以上の動きを持つモーションはエラーとして除外する
    max_velocity: int = 20

    # Motion-centroid-correctionを行うかどうか
    enable_motion_correction: bool = True

    # Motion-centroid-correctionでテンプレートマッチの結果にかける円フィルタの半径」
    # もとのキーフレームの矩形の大きさ//2に対する比率
    centroid_correction_template_match_filter_radius_ratio: float = 0.5


def _check_dtype_and_shape(dtype, shape):
    def checker(a):
        if not isinstance(dtype, tuple):
            dtype_normalized = dtype,
        else:
            dtype_normalized = dtype

        if a.dtype not in dtype_normalized:
            raise TypeError(f'expected {dtype=}, provided {a.dtype=}')

        if a.ndim != len(shape):
            raise TypeError(f'expected ndim={len(shape)}, provided {a.ndim=}')

        for i in range(len(shape)):
            if shape[i] is None:
                continue
            if shape[i] != a.shape[i]:
                raise TypeError(
                    f'expected {shape=}, {shape[i]=} for dimension #{i}, '
                    f'provided array with {a.shape=}, whose size of dimension #{i} is {a.shape[i]}'
                )

    return checker


class PMDetectorInputTimeSeriesEntry:
    @property
    def original_image(self):
        return self.__original_image

    @property
    def diff_image(self):
        return self.__diff_image

    @property
    def timestamp(self):
        return self.__timestamp

    @property
    def height(self):
        # returns the frame height sampled from `self.frame_original`
        return self.original_image.shape[0]

    @property
    def width(self):
        # returns the frame width sampled from `self.frame_original`
        return self.original_image.shape[1]

    @property
    def frame_shape(self):
        return self.height, self.width, 3

    @property
    def _checker_for_frame_image(self):
        return _check_dtype_and_shape(
            dtype=np.uint8,
            shape=self.frame_shape
        )

    def __init__(self, original_image, diff_image, timestamp):
        """
        PMDetectorの入力データのうちの時系列データ
        :param original_image: np.uint8 with shape(height, width, channels)
        :param diff_image: np.uint8 with shape(height, width, channels)
        :param timestamp: float
        """
        # assign members
        self.__original_image = original_image
        self.__diff_image = diff_image
        self.__timestamp = timestamp

        # check arguments
        self._checker_for_frame_image(original_image)
        self._checker_for_frame_image(diff_image)

        assert isinstance(timestamp, float), type(timestamp)


class PMDetectorInput:
    @property
    def target_frame(self):
        return self.__target_frame

    @property
    def next_frame(self):
        return self.__next_frame

    @property
    def rect_normalized(self):
        return self.__rect_normalized

    @property
    def rect_actual_scaled(self):
        return self.__rect_normalized.to_actual_scaled(
            width=self.width,
            height=self.height
        )

    @property
    def height(self):
        # returns the frame height sampled from `self.target_frame`
        return self.target_frame.height

    @property
    def width(self):
        # returns the frame width sampled from `self.target_frame`
        return self.target_frame.width

    @property
    def frame_shape(self):
        return self.target_frame.frame_shape

    def __init__(self, target_frame, next_frame, detection_rect_normalized):
        # noinspection GrazieInspection
        """
        PMDetectorの入力データ

        :param target_frame: PMDetectorInputTimeSeriesEntry
        :param next_frame: PMDetectorInputTimeSeriesEntry
        :param detection_rect_normalized: RectNormalized
        """
        # assign members
        self.__target_frame = target_frame
        self.__next_frame = next_frame
        self.__rect_normalized = detection_rect_normalized

        # check arguments
        assert isinstance(target_frame, PMDetectorInputTimeSeriesEntry), target_frame
        assert isinstance(next_frame, PMDetectorInputTimeSeriesEntry), next_frame
        assert target_frame.frame_shape == next_frame.frame_shape, \
            (target_frame.frame_shape, next_frame.frame_shape)
        assert isinstance(detection_rect_normalized, RectNormalized), detection_rect_normalized


class OneWriteManyReadDescriptor:
    def __init__(self, default):
        self.__value = default
        self.__value_set = False

    def freeze(self):
        self.__value_set = True

    def __get__(self, obj, owner=None):
        if not self.__value_set:
            raise ValueError('attribute not set')
        return self.__value

    def __set__(self, obj, value):
        if self.__value_set:
            raise ValueError('attribute is read-only; only one write to attribute allowed')
        self.__value = value
        self.freeze()


class PMDetectorResult:
    original_images_clipped: np.ndarray
    diff_images_clipped: np.ndarray
    keypoints: list[np.ndarray]
    keyframes: list[np.ndarray]
    match_index_pair: np.ndarray
    distance_matrix: np.ndarray
    local_centroid: np.ndarray  # centroids in rect coordinate

    def __init__(self):
        self.__field_names = type(self).__annotations__.keys()
        self.__descriptors = []
        for field_name in self.__field_names:
            field_accessor = OneWriteManyReadDescriptor(default=None)
            self.__descriptors.append(field_accessor)
            setattr(self, field_name, field_accessor)

    def _freeze_fields(self):
        for accessor in self.__descriptors:
            accessor.freeze()

    @functools.cached_property
    def contains_keypoints(self):
        assert self.keypoints is not None
        return all(keypoints.size > 0 for keypoints in self.keypoints)

    @functools.cached_property
    def n_matches(self):
        assert self.match_index_pair is not None
        return len(self.match_index_pair.T)


class PMComputer:
    def __init__(self, parameter: PMDetectorParameter, source: PMDetectorInput):
        self._p = parameter
        self._input = source
        self._result = PMDetectorResult()

    def __extract_key_frame_around(self, image, index_axis_1, index_axis_2):
        assert image.ndim == 3, image.shape

        # decide the size of padding
        size = self._p.key_image_size

        # pad source image
        # noinspection PyTypeChecker
        padded_image = np.pad(
            image,
            ((size, size), (size, size), (0, 0)),
            constant_values=0
        )

        # extract key frames
        keyframes = []
        # for each provided point
        for idx1, idx2 in zip(index_axis_1, index_axis_2):
            # shift center position by padding size
            c_idx1, c_idx2 = idx1 + size, idx2 + size
            # create index slice
            keyframe_index = (
                slice(c_idx1 - size, c_idx1 + size + 1),
                slice(c_idx2 - size, c_idx2 + size + 1),
                slice(None, None)
            )
            # extract key frame around (idx1, idx2)
            keyframe = padded_image[keyframe_index]
            # append keyframe
            keyframes.append(keyframe)

        # returns keyframes by np-array if any keyframes found; otherwise returns None
        if keyframes:
            return np.stack(keyframes)
        else:
            return None

    def __array_checkers_for_detect_keypoints(self):
        height = self._input.height
        width = self._input.width
        rect_height = self._input.rect_actual_scaled.size.y
        rect_width = self._input.rect_actual_scaled.size.x

        @dataclass(frozen=True)
        class ArrayCheckersForDetectKeypoints:
            ui8_2hw3 = _check_dtype_and_shape(
                dtype=np.uint8,
                shape=(2, height, width, 3)
            )
            f32_2hw3_clipped = _check_dtype_and_shape(
                dtype=np.float32,
                shape=(2, rect_height, rect_width, 3)
            )
            ui8_2hw3_clipped = _check_dtype_and_shape(
                dtype=np.uint8,
                shape=(2, rect_height, rect_width, 3)
            )
            f32_2hw_clipped = _check_dtype_and_shape(
                dtype=np.float32,
                shape=(2, rect_height, rect_width)
            )

        return ArrayCheckersForDetectKeypoints

    def detect_keypoints(self):
        # diff -> _process_input -> _process_mean -> _local_max -> <x>
        # original -> _process_input -> <x>
        #   <x> -> extract_frames_around

        check = self.__array_checkers_for_detect_keypoints()

        # ** prepare for rect

        # generate clip index (time-axis, y-axis(height), x-axis(width), channel-axis)
        detection_region_clip_index \
            = slice(None, None), *self._input.rect_actual_scaled.index3d

        # *** focus on the original frame images

        # stack original images along time-axis
        original_images = np.stack([
            self._input.target_frame.original_image,
            self._input.next_frame.original_image
        ])
        check.ui8_2hw3(original_images)

        # clip by detection region
        original_images = original_images[detection_region_clip_index]
        check.ui8_2hw3_clipped(original_images)

        # convert rgb-values from uint8 to normalized float
        original_images = original_images.astype(np.float32) / 256.0
        assert 0 <= original_images.min() and original_images.max() < 1, \
            (original_images.min(), original_images.max())
        check.f32_2hw3_clipped(original_images)

        # *** focus on the diff images

        # stack diff images along time-axis
        diff_images = np.stack([
            self._input.target_frame.diff_image,
            self._input.next_frame.diff_image
        ])
        check.ui8_2hw3(diff_images)

        # clip by detection region
        diff_images = diff_images[detection_region_clip_index]
        check.ui8_2hw3_clipped(diff_images)

        # convert rgb-values from uint8 to normalized float
        diff_images = diff_images.astype(np.float32) / 256.0
        assert 0 <= diff_images.min() and diff_images.max() < 1, \
            (diff_images.min(), diff_images.max())
        check.f32_2hw3_clipped(diff_images)

        # convert rgb channels to grayscale
        diff_images = diff_images.mean(axis=-1)
        check.f32_2hw_clipped(diff_images)

        # remove luminance except significant ones for each frame time
        for i in range(2):
            diff_images[i] = np.where(
                diff_images[i] < np.percentile(diff_images[i], 95),
                0,
                diff_images[i]
            )
            check.f32_2hw_clipped(diff_images)

        # generate mean filter matrix
        filter_shape = np.array(
            self._input.rect_actual_scaled.size) // self._p.mean_filter_size_factor
        filter_matrix = np.ones(filter_shape, dtype=np.float32)
        filter_matrix /= filter_matrix.sum()
        assert np.isclose(filter_matrix.sum(), 1), filter_matrix.sum()

        # apply filter for each frame time
        for i in range(2):
            # noinspection PyUnresolvedReferences
            diff_images[i] = scipy.ndimage.convolve(
                diff_images[i],
                weights=filter_matrix,
                mode='constant'
            )

        # calculate local maxima for each frame time
        keypoints = [None, None]
        for i in range(2):
            keypoints[i] = skimage.feature.peak_local_max(
                diff_images[i],
                min_distance=max(diff_images[i].shape) // self._p.local_max_distance_factor
            )

            # peak_local_max() returns the local maximum point in the order of
            # (height, width), which match our expectation
            # (1st-axis, 2nd-axis) == (y-axis, x-axis).
            keypoints[i] = keypoints[i]

            _check_dtype_and_shape(
                dtype=np.int64,
                shape=(None, 2)  # (<number of local maxima>, [x, y])
            )(keypoints[i])

        # *** extract key frames
        # extract key frames for each frame time
        keyframes = [None, None]
        for i in range(2):
            keyframes[i] = self.__extract_key_frame_around(
                image=original_images[i],
                index_axis_1=keypoints[i][:, 0],
                index_axis_2=keypoints[i][:, 1]
            )
            _check_dtype_and_shape(
                dtype=np.float32,
                shape=(
                    None,
                    self._p.key_image_size * 2 + 1,
                    self._p.key_image_size * 2 + 1,
                    3
                )
            )(keyframes[i])

        # *** set the whole result in this section
        check.f32_2hw3_clipped(original_images)
        self._result.original_images_clipped = original_images
        check.f32_2hw_clipped(diff_images)
        self._result.diff_images_clipped = diff_images
        # TODO: check for keypoints
        self._result.keypoints = keypoints
        # TODO: check for keyframes
        self._result.keyframes = keyframes

    @classmethod
    def _split_single_channel_image_3x3(cls, image) -> list[np.ndarray]:
        assert image.ndim == 2, image.shape
        n = image.shape[0] // 3  # assuming t.shape[0] == t.shape[1]
        m = n * 2
        slices = slice(None, n), slice(n, m), slice(m, None)
        splits = [image[s1, s2] for s1 in slices for s2 in slices]  # flatten
        return splits

    @classmethod
    def _split_image_3x3(cls, image) -> list[list[np.ndarray]]:
        assert image.ndim == 3 and image.shape[-1] == 3, image.shape
        return [
            cls._split_single_channel_image_3x3(image[:, :, ch])
            for ch in range(3)  # extract each channel (R, G, B) of `image`
        ]

    @classmethod
    def _extract_gray_distribution_feature(cls, image, n_bins):
        assert image.ndim == 2, image.shape

        # this method is faster than np.histogram()

        # convert image to integer of [0, n_bins)
        digitized_image = (image * n_bins).astype(np.int8)
        assert np.all((0 <= digitized_image) & (digitized_image < n_bins))

        # count each value
        values, counts = np.unique(digitized_image, return_counts=True)

        # make histogram
        hist = np.zeros(n_bins)
        hist[values] = counts

        return hist

    @classmethod
    def _extract_feature_for_matching(cls, image):
        assert image.ndim == 3 and image.shape[-1] == 3, image.shape
        image_split = cls._split_image_3x3(image)
        feature = np.concatenate([
            cls._extract_gray_distribution_feature(p, n_bins=16)
            for channel_split in image_split
            for p in channel_split
        ])
        feature_normalized = feature / feature.sum()
        return feature_normalized

    def _find_mutual_best_match(self, dist_mat):
        # distance matrix has 2 dimensions, 1st dim and 2nd dim
        assert dist_mat.ndim == 2, dist_mat.shape

        # generate index vector of 1st dim and 2nd dim
        index_dim1 = np.arange(dist_mat.shape[0])
        index_dim2 = np.arange(dist_mat.shape[1])

        # find one-sided love
        best_forward = dist_mat.argmin(axis=1)  # dist_mat 1st dim to 2nd dim
        best_backward = dist_mat.argmin(axis=0)  # dist_mat 2nd dim to 1st dim

        # find mutual love
        mutual_love_dim1to2 = best_backward[best_forward] == index_dim1
        best_index_dim1to2 = index_dim1[mutual_love_dim1to2]
        best_index_dim2to1 = index_dim2[best_forward[mutual_love_dim1to2]]
        assert best_index_dim1to2.shape == best_index_dim2to1.shape, \
            (best_index_dim1to2.shape, best_index_dim2to1.shape)

        # check the distance is better than global parameter `mutual_match_max_cos_distance`
        distance = dist_mat[best_index_dim1to2, best_index_dim2to1]
        mask = distance < self._p.mutual_match_max_cos_distance
        best_index_dim1to2 = best_index_dim1to2[mask]
        best_index_dim2to1 = best_index_dim2to1[mask]

        return np.stack([best_index_dim1to2, best_index_dim2to1])

    def extract_matches(self):
        keyframes = self._result.keyframes
        assert keyframes is not None

        # extract feature vectors of key-frames for matching
        features = [
            list(map(self._extract_feature_for_matching, keyframes[0])),
            list(map(self._extract_feature_for_matching, keyframes[1]))
        ]

        # calculate distance matrix with the extracted feature vectors above
        dist_mat_cos = cosine_similarity(*features)
        dist_mat = 1 - np.clip(dist_mat_cos, a_min=0, a_max=1)

        # find mutual match
        match_index_pair = self._find_mutual_best_match(dist_mat)

        self._result.match_index_pair = match_index_pair
        self._result.distance_matrix = dist_mat

    _FILTER_GRAD_FIRST_AXIS = np.array([
        [0, 0, 0],
        [+0.5, 0, -0.5],
        [0, 0, 0]
    ])
    _FILTER_GRAD_SECOND_AXIS = _FILTER_GRAD_FIRST_AXIS.T

    def _correct_motion_centroid(self, keyframe_src: np.ndarray, keyframe_dst: np.ndarray):
        keyframe_pair = [keyframe_src, keyframe_dst]

        # This method does not produces valuable difference
        # process keyframes
        # for i in range(2):
        #     keyframe = keyframe_pair[i]
        #
        #     # convert to grayscale
        #     keyframe_gray = keyframe.mean(axis=-1)
        #
        #     # noinspection PyUnresolvedReferences
        #     grad1 = scipy.ndimage.convolve(
        #         keyframe_gray,
        #         self._FILTER_GRAD_FIRST_AXIS,
        #         mode='nearest'
        #     )
        #     # noinspection PyUnresolvedReferences
        #     grad2 = scipy.ndimage.convolve(
        #         keyframe_gray,
        #         self._FILTER_GRAD_SECOND_AXIS,
        #         mode='nearest'
        #     )
        #
        #     # take geometric mean of vertical grad and horizontal grad
        #     # TODO: proof of this method: take geometric mean of vertical grad and horizontal grad
        #     gard_composite = np.sqrt(np.square(grad1) + np.square(grad2))
        #
        #     # take local mean of `gard_composite` image
        #     gard_composite = skimage.filters.rank.mean(
        #         skimage.util.img_as_ubyte(gard_composite),
        #         np.ones((3, 3))
        #     )
        #
        #     # filter keyframe by brighter pixels of `gard_composite`
        #     mask = gard_composite > np.percentile(gard_composite, 50)
        #     keyframe = np.where(np.tile(mask[..., None], 3), keyframe, 0)
        #
        #     keyframe_pair[i] = keyframe

        # template match small regions of keyframes
        # noinspection PyTypeChecker
        tp: np.ndarray = skimage.feature.match_template(
            image=keyframe_pair[1],
            template=keyframe_pair[0],
            pad_input=True,
            mode='reflect'
        )[:, :, 0]  # template match returns 3-dimensional array of same values

        # apply radius filter on match result
        old_centroid = np.array(keyframe_dst.shape)[:-1] // 2
        x, y = np.meshgrid(np.arange(tp.shape[0]), np.arange(tp.shape[1]))
        x, y = x - old_centroid[0], y - old_centroid[1]
        rr = self._p.centroid_correction_template_match_filter_radius_ratio
        r = int(tp.shape[0] * rr) // 2
        tp[x * x + y * y > r * r] = 0

        # extract best template match
        mask_max = tp == tp.max()
        if np.count_nonzero(mask_max) == 1:
            xs, ys = np.where(mask_max)
            new_centroid = np.array([xs[0], ys[0]])
            correction = new_centroid - old_centroid
        else:
            # if no match exists, perform no correction
            correction = np.array([0, 0])

        # plt.figure()
        # plt.subplot(311)
        # plt.imshow(keyframe_src)
        # plt.scatter(keyframe_src.shape[0] // 2, keyframe_src.shape[1] // 2, color='yellow',
        #             marker='x', s=300)
        # plt.subplot(312)
        # plt.imshow(keyframe_dst)
        # plt.scatter([ys[0]], [xs[0]], color='yellow', marker='x', s=300)
        # plt.subplot(313)
        # plt.imshow(tp)
        # plt.show()
        # import time
        # time.sleep(0.1)

        return correction

    def generate_motion_centroids(self):
        local_centroid = np.array([
            [
                self._result.keypoints[i][match_index]
                for match_index in self._result.match_index_pair[i]
            ] for i in range(2)
        ])

        _check_dtype_and_shape(
            dtype=np.int64,
            shape=(2, None, 2)
        )

        if self._p.enable_motion_correction:
            corrections_for_each_match = []
            for match_index_pair in self._result.match_index_pair.T:
                keyframes = [self._result.keyframes[i][match_index_pair[i]] for i in range(2)]
                correction = self._correct_motion_centroid(*keyframes)
                corrections_for_each_match.append(correction)
            corrections_for_each_match = np.stack(corrections_for_each_match)

            # correct local center of second frame
            local_centroid[1] += corrections_for_each_match

        self._result.local_centroid = local_centroid

    def compute(self) -> PMDetectorResult:
        self.detect_keypoints()
        if self._result.contains_keypoints:
            self.extract_matches()
            self.generate_motion_centroids()
        # noinspection PyProtectedMember
        self._result._freeze_fields()
        return self._result


class PMDetectorTester:
    @staticmethod
    def test_detect_keypoints(result: PMDetectorResult):
        plt.figure(figsize=(16, 8))
        for i in range(2):
            plt.subplot(120 + i + 1)
            plt.imshow(result.original_images_clipped[i])
            plt.scatter(
                *result.keypoints[i].T[::-1],
                c='yellow',
                marker='x',
                s=500,
                linewidths=3
            )
        plt.show()
        pprint(result.keypoints)

    @staticmethod
    def test_matches(result: PMDetectorResult):
        matches_tuple = {tuple(x) for x in result.match_index_pair.T}
        n_keys_a, n_keys_b = map(len, result.keyframes)
        fig, axes = plt.subplots(n_keys_a + 2, n_keys_b + 2, figsize=(40, 40))

        from tqdm import tqdm

        for i in tqdm(range(n_keys_a)):
            for j in range(n_keys_b):
                axes[i + 2, j + 2].bar([0], [result.distance_matrix[i, j]])
                axes[i + 2, j + 2].set_ylim(0, 1)
                if (i, j) in matches_tuple:
                    axes[i + 2, j + 2].scatter([0], [0.5], color='red', s=500)

        for i in range(n_keys_a):
            axes[i + 2, 0].imshow(result.original_images_clipped[0])
            axes[i + 2, 0].scatter(
                result.keypoints[0][i, 1],
                result.keypoints[0][i, 0],
                color='yellow',
                marker='x',
                s=200
            )
            axes[i + 2, 1].imshow(result.keyframes[0][i])
        for i in range(n_keys_b):
            axes[0, i + 2].imshow(result.original_images_clipped[1])
            axes[0, i + 2].scatter(
                result.keypoints[1][i, 1],
                result.keypoints[1][i, 0],
                color='yellow',
                marker='x',
                s=200
            )
            axes[1, i + 2].imshow(result.keyframes[1][i])

        for ax in axes.flatten():
            ax.axis('off')
        fig.tight_layout()
        fig.savefig('local_max_feature_dist_mat.jpg')
        fig.show()

    @staticmethod
    def test_local_centroids(result: PMDetectorResult):
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(result.original_images_clipped[0])
        for i, mi in enumerate(result.match_index_pair[0]):
            plt.scatter(
                *result.local_centroid[0][i][::-1],
                color='yellow',
                marker='x',
                s=200
            )
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(result.original_images_clipped[1])
        for i, mi in enumerate(result.match_index_pair[1]):
            plt.scatter(
                *result.local_centroid[1][i][::-1],
                color='yellow',
                marker='x',
                s=200
            )
        plt.axis('off')

        plt.tight_layout()
        plt.show()


class PMDetector:
    def __init__(self, parameter: PMDetectorParameter):
        self.__parameter = parameter

    def compute(self, source: PMDetectorInput):
        return PMComputer(self.__parameter, source).compute()


if __name__ == '__main__':
    def main():
        video_name = '20230205_04_Narumoto_Harimoto'

        with storage.create_instance(
                domain='numpy_storage',
                entity=video_name,
                context='frames',
                mode='r',
        ) as snp_video_frame:
            assert isinstance(snp_video_frame, snp.NumpyStorage)

            i = 191

            snp_entry_target = snp_video_frame.get_entry(i)
            snp_entry_next = snp_video_frame.get_entry(i + 1)
            assert isinstance(snp_entry_target, snp_context.SNPEntryVideoFrame)
            assert isinstance(snp_entry_next, snp_context.SNPEntryVideoFrame)

            detector: Optional[PMDetector] = PMDetector(
                PMDetectorParameter(enable_motion_correction=True)
            )
            result = detector.compute(
                PMDetectorInput(
                    target_frame=PMDetectorInputTimeSeriesEntry(
                        original_image=snp_entry_target.original,
                        diff_image=snp_entry_target.motion,
                        timestamp=float(snp_entry_target.timestamp)
                    ),
                    next_frame=PMDetectorInputTimeSeriesEntry(
                        original_image=snp_entry_next.original,
                        diff_image=snp_entry_next.motion,
                        timestamp=float(snp_entry_next.timestamp)
                    ),
                    detection_rect_normalized=train_input.frame_rects.normalized(video_name)
                )
            )

            # PMDetectorTester.test_detect_keypoints(result)
            # PMDetectorTester.test_matches(result)
            PMDetectorTester.test_local_centroids(result)

            # TODO: normalize result


    main()
