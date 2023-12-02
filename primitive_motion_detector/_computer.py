from dataclasses import dataclass

import numpy as np
import scipy.ndimage
import skimage.feature
from sklearn.metrics.pairwise import cosine_similarity

from ._parameter import PMDetectorParameter
from ._result import PMDetectorResult
from ._source import PMDetectorSource
from ._util import check_dtype_and_shape


class _PMComputerStubs:
    _p: PMDetectorParameter
    _input: PMDetectorSource
    _result: PMDetectorResult


class _PMComputerKeyFramerDetectorMixin(_PMComputerStubs):
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
            ui8_2hw3 = check_dtype_and_shape(
                dtype=np.uint8,
                shape=(2, height, width, 3)
            )
            f32_2hw3_clipped = check_dtype_and_shape(
                dtype=np.float32,
                shape=(2, rect_height, rect_width, 3)
            )
            ui8_2hw3_clipped = check_dtype_and_shape(
                dtype=np.uint8,
                shape=(2, rect_height, rect_width, 3)
            )
            f32_2hw_clipped = check_dtype_and_shape(
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
            )  # [N_KEYPOINTS, (Y, X)]

            # peak_local_max() returns the local maximum point in the order of
            # (height, width), which match our expectation
            # (1st-axis, 2nd-axis) == (y-axis, x-axis).
            keypoints[i] = keypoints[i]

            check_dtype_and_shape(
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
            check_dtype_and_shape(
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


class _PMComputerMatchExtractorMixin(_PMComputerStubs):
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


class _PMComputerMotionCentroidGenerator(_PMComputerStubs):
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

        # extract the best template match
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

        check_dtype_and_shape(
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

        local_centroid_normalized = np.zeros_like(local_centroid, dtype=np.float32)

        # normalize by rect
        normalizer = self._input.rect_actual_scaled.normalize_points_inside_based_on_corner
        for i in range(2):
            local_centroid_normalized[i] = normalizer(local_centroid[i][:, ::-1])[:, ::-1]

        # set result
        check_dtype_and_shape(
            dtype=np.int64,
            shape=(2, None, 2)
        )(local_centroid)
        self._result.local_centroid = local_centroid

        global_centroid = local_centroid + self._input.rect_actual_scaled.p_min[::-1]
        check_dtype_and_shape(
            dtype=np.int64,
            shape=(2, None, 2)
        )(global_centroid)
        self._result.global_centroid = global_centroid

        check_dtype_and_shape(
            dtype=np.float32,
            shape=(2, None, 2)
        )(local_centroid_normalized)
        self._result.local_centroid_normalized = local_centroid_normalized


class PMComputer(
    _PMComputerKeyFramerDetectorMixin,
    _PMComputerMatchExtractorMixin,
    _PMComputerMotionCentroidGenerator
):
    def __init__(self, parameter: PMDetectorParameter, source: PMDetectorSource):
        self._p = parameter
        self._input = source
        self._result = PMDetectorResult()

    def compute(self) -> PMDetectorResult:
        self.detect_keypoints()
        if self._result.contains_keypoints:
            self.extract_matches()
            self.generate_motion_centroids()
        # noinspection PyProtectedMember
        self._result._freeze_fields()
        return self._result
