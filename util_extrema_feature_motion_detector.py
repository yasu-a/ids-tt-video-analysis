from dataclasses import dataclass
from pprint import pprint
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage
import skimage.feature

import npstorage_context as snp_context
import storage
import storage.npstorage as snp
import train_input
from train_input import RectNormalized


@dataclass(frozen=True)
class PMDetectorParameter:
    # 与えられた画像にかける平均値フィルタの大きさ`画像サイズ // mean_filter_size_factor`
    mean_filter_size_factor = 32

    # 輝度極大点を求めるときに`skimage.feature.peak_local_max`の引数`min_distance`に
    # 与える値`画像サイズ // local_max_distance_factor`
    local_max_distance_factor = 32

    # 輝度極大点を求めるときに極大点の対象とする最小輝度
    motion_local_max_thresh = 0.03

    # キーフレームの相互マッチングのときに，cos距離がこの値以下のマッチングが対象となる。
    # cos距離は`1 - np.clip(cos_distance, 0, 1)`で算出され，0に近いほうが距離が近い。
    mutual_match_max_cos_distance = 0.3

    # 切り出すキーフレームの大きさで，中心点から±`key_image_size`の範囲が切り出される。
    # 実際に切り出されるキーフレームの大きさは`key_image_size * 2 + 1`
    key_image_size: int = 32

    # これ以上の動きを持つモーションはエラーとして除外する
    max_velocity = 20

    # Motion-centroid-correctionを行うかどうか
    enable_motion_correction = True


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


class PMDetectorResult:
    original_images_clipped: np.ndarray = None
    diff_images_clipped: np.ndarray = None
    keypoints: list[np.ndarray] = None
    keyframes: list[np.ndarray] = None


class PMComputer:
    def __init__(self, parameter: PMDetectorParameter, source: PMDetectorInput):
        self.__p = parameter
        self.__src = source
        self.__result = PMDetectorResult()

    def __extract_key_frame_around(self, image, index_axis_1, index_axis_2):
        assert image.ndim == 3, image.shape

        # decide the size of padding
        size = self.__p.key_image_size

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
        height = self.__src.height
        width = self.__src.width
        rect_height = self.__src.rect_actual_scaled.size.y
        rect_width = self.__src.rect_actual_scaled.size.x

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
            = slice(None, None), *self.__src.rect_actual_scaled.index3d

        # *** focus on the original frame images

        # stack original images along time-axis
        original_images = np.stack([
            self.__src.target_frame.original_image,
            self.__src.next_frame.original_image
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
            self.__src.target_frame.diff_image,
            self.__src.next_frame.diff_image
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
            self.__src.rect_actual_scaled.size) // self.__p.mean_filter_size_factor
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
                min_distance=max(diff_images[i].shape) // self.__p.local_max_distance_factor
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
                    self.__p.key_image_size * 2 + 1,
                    self.__p.key_image_size * 2 + 1,
                    3
                )
            )(keyframes[i])

        # *** set the whole result in this section
        check.f32_2hw3_clipped(original_images)
        self.__result.original_images_clipped = original_images
        check.f32_2hw_clipped(diff_images)
        self.__result.diff_images_clipped = diff_images
        # TODO: check for keypoints
        self.__result.keypoints = keypoints
        # TODO: check for keyframes
        self.__result.keyframes = keyframes

    @classmethod
    def _split_3x3(cls, a):
        n = a.shape[0] // 3  # assuming t.shape[0] == t.shape[1]
        m = n * 2
        slices = slice(None, n), slice(n, m), slice(m, None)
        splits = [a[s1, s2] for s1 in slices for s2 in slices]
        return splits

    @classmethod
    def _extract_match_feature(cls, image):
        assert image.ndim == 3 and image.shape[-1] == 3, image.shape
        feature = []
        for ch in range(3):
            image_single_ch = image[:, :, ch]  # extract each channel (R, G, B) of `image`
            for p in cls._split_3x3(image_single_ch):
                hist, _ = np.histogram(p, bins=16, range=(0, 1))
                feature.append(hist)
        total_feature = np.concatenate(feature)
        return total_feature / total_feature.sum()

    def extract_matches(self):
        keyframes = self.__result.keyframes
        assert keyframes is not None

        features = [
            list(map(self._extract_match_feature, keyframes[0])),
            list(map(self._extract_match_feature, keyframes[1]))
        ]

        print(features[0][0])
        pass

    def compute(self) -> PMDetectorResult:
        self.detect_keypoints()
        self.extract_matches()
        return self.__result


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

            i = 180

            snp_entry_target = snp_video_frame.get_entry(i)
            snp_entry_next = snp_video_frame.get_entry(i + 1)
            assert isinstance(snp_entry_target, snp_context.SNPEntryVideoFrame)
            assert isinstance(snp_entry_next, snp_context.SNPEntryVideoFrame)

            detector: Optional[PMDetector] = PMDetector(
                PMDetectorParameter()
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


    main()
