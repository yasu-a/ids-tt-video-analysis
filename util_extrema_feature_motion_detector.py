import functools

import numpy as np
import scipy.ndimage
import skimage.feature
from sklearn.metrics.pairwise import cosine_similarity

import util


class ExtremaFeatureMotionDetector:
    def __init__(
            self,
            detection_region_rect,  # list of [height:slice, width:slice]
            mean_conv_win_size_factor=32,
            motion_local_max_distance_factor=32,
            motion_local_max_thresh=0.03,
            mutual_match_max_cos_distance=0.3,  # cos_distance = 1 - np.clip(cos_dist, 0, 1)
            key_image_size=32,
            max_velocity=20
    ):
        self.__p_detection_region_rect = detection_region_rect
        self.__c_rect = *self.__p_detection_region_rect, slice(None, None)
        self.__p_mean_conv_win_size_factor = mean_conv_win_size_factor
        self.__p_motion_local_max_distance_factor = motion_local_max_distance_factor
        self.__p_motion_local_max_thresh = motion_local_max_thresh
        self.__p_mutual_match_max_cos_distance = mutual_match_max_cos_distance
        self.__p_key_image_size = key_image_size // 2
        self.__p_max_velocity = max_velocity

    def _process_input(self, img):
        return img[self.__c_rect].astype(np.float32) / 256.0

    @functools.cache
    def _mean_conv_filter(self, image_shape):
        fil_shape = np.array(image_shape) // self.__p_mean_conv_win_size_factor
        fil = np.ones(fil_shape, dtype=np.float32)
        fil = fil / fil.sum()
        return fil

    def _process_mean(self, motion_image):
        # return scipy.ndimage.median_filter(
        #     motion_image,
        #     size=np.array(motion_image.shape) // self.__p_mean_conv_win_size_factor,
        #     mode='constant'
        # )
        motion_image = np.where(
            motion_image < np.percentile(motion_image, 95),
            0,
            motion_image
        )
        return scipy.ndimage.convolve(
            motion_image,
            weights=self._mean_conv_filter(motion_image.shape),
            mode='constant',
        )

    def _local_max(self, img):
        points = skimage.feature.peak_local_max(
            img,
            min_distance=max(img.shape) // self.__p_motion_local_max_distance_factor
        )
        return points[img[tuple(points.T)] > self.__p_motion_local_max_thresh]

    @classmethod
    def _split_3x3(cls, a):
        n = a.shape[0] // 3  # assuming t.shape[0] == t.shape[1]
        m = n * 2
        slices = slice(None, n), slice(n, m), slice(m, None)
        splits = [a[s1, s2] for s1 in slices for s2 in slices]
        return splits

    @classmethod
    def _extract_feature(cls, img):
        r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
        feature = []
        for t in [r, g, b]:
            for p in cls._split_3x3(t):
                hist, _ = np.histogram(p, bins=16, range=(0, 1))
                feature.append(hist)
        total_feature = np.concatenate(feature)
        return total_feature / total_feature.sum()

    def _compare_key_batches(self, batch_lst_a, batch_lst_b):
        feature_a = list(map(self._extract_feature, batch_lst_a))
        feature_b = list(map(self._extract_feature, batch_lst_b))

        dist_mat = 1 - np.clip(cosine_similarity(feature_a, feature_b), 0, 1)

        return dist_mat

    def _find_mutual_best_match(self, dist_mat):
        best_forward = dist_mat.argmin(axis=1)  # x to y (1st dim to 2nd dim)
        x_index = np.arange(dist_mat.shape[0])
        best_backward = dist_mat.argmin(axis=0)  # y to x
        y_index = np.arange(dist_mat.shape[1])

        mutual_love_from_forward = best_backward[best_forward] == x_index
        x = x_index[mutual_love_from_forward]
        y = y_index[best_forward[mutual_love_from_forward]]

        assert x.size == y.size, (x.shape, y.shape)

        mask = dist_mat[x, y] < self.__p_mutual_match_max_cos_distance
        x, y = x[mask], y[mask]

        return np.stack([x, y]).T

    def _extract_key_points(self, original_images, motion_images):
        class Keys(dict):
            POINT_SPECIFIC_FEATURE_NAMES = 'local_center', 'global_center', 'frame'

            def contains_no_key_points(self):
                return not self['frame_a'].size or not self['frame_b'].size

        def process():
            # motion_x: npa[height, width, channels]
            motion_a, motion_b = motion_images
            # original_x: npa[height, width, channels]
            original_a, original_b = original_images

            # motion_x: npa[rect_height, rect_width]
            motion_a = self._process_input(motion_a).mean(axis=2)
            motion_b = self._process_input(motion_b).mean(axis=2)

            # original_x: npa[rect_height, rect_width, channels]
            original_a = self._process_input(original_a)
            original_b = self._process_input(original_b)

            # motion_x_mean: npa[rect_height, rect_width]
            motion_a_mean = self._process_mean(motion_a)
            motion_b_mean = self._process_mean(motion_b)

            # motion_x_local_max: npa[N_MAX_x, 2(2nd axis, 1st axis)]
            motion_a_local_max = self._local_max(motion_a_mean)
            motion_b_local_max = self._local_max(motion_b_mean)

            # key_img_x: npa[N_MAX_x, frame_size, frame_size, channels]
            key_img_a = util.extract_frames_around(
                original_a,
                x=motion_a_local_max[:, 1],
                y=motion_a_local_max[:, 0],
                size=self.__p_key_image_size
            )
            key_img_b = util.extract_frames_around(
                original_b,
                x=motion_b_local_max[:, 1],
                y=motion_b_local_max[:, 0],
                size=self.__p_key_image_size
            )

            rect_offset = self.__c_rect[0].start, self.__c_rect[1].start

            return Keys(
                valid=False,  # holds unmatched points
                motion_a=motion_a,
                motion_b=motion_b,
                original_a=original_a,
                original_b=original_b,
                local_center_a=motion_a_local_max,  # key point
                local_center_b=motion_b_local_max,  # key point
                global_center_a=motion_a_local_max + rect_offset,  # key point
                global_center_b=motion_b_local_max + rect_offset,  # key point
                frame_a=key_img_a,  # key frame
                frame_b=key_img_b,  # key frame
                count_a=len(key_img_a),
                count_b=len(key_img_b)
            )

        return process()

    def _extract_matches(self, keys_):
        class Matches(dict):
            def __generate_pairs(self):
                for i, t in enumerate('ab'):
                    for n in self.keys.POINT_SPECIFIC_FEATURE_NAMES:
                        field_name = f'{n}_{t}'
                        match_index = self.match_index_pair[:, i]
                        self[field_name] = self.keys[field_name][match_index]

                self['match_index_a'] = self.match_index_pair[:, 0]
                self['match_index_b'] = self.match_index_pair[:, 1]

                self['original_a'] = self.keys['original_a']
                self['original_b'] = self.keys['original_b']
                self['motion_a'] = self.keys['motion_a']
                self['motion_b'] = self.keys['motion_b']

            def __generate_additional_data(self):
                self['velocity'] = self['local_center_b'] - self['local_center_a']
                self['velocity_x'] = self['velocity'][:, 0]
                self['velocity_y'] = self['velocity'][:, 1]
                self['velocity_norm'] = np.linalg.norm(self['velocity'], axis=1)

            def __init__(self, *, keys, match_index_pair, dist_mat):
                super().__init__()
                self.keys = keys
                self.match_index_pair = match_index_pair
                self.dist_mat = dist_mat
                self.__generate_pairs()
                self.__generate_additional_data()

            def apply_filter(self, mask):
                for k in self:
                    if k.startswith('original') or k.startswith('motion'):
                        continue
                    self[k] = self[k][mask]

        def process(keys):
            if keys.contains_no_key_points():
                return keys | dict(valid=False)  # no motion detected

            # dist_mat: npa[N_MAX_a, N_MAX_b]
            dist_mat = self._compare_key_batches(keys['frame_a'], keys['frame_b'])
            # matches: npa[N_MATCH, 2(indexes of a, indexes of b)]
            match_index_pair = self._find_mutual_best_match(dist_mat)

            matches = Matches(
                keys=keys,
                match_index_pair=match_index_pair,
                dist_mat=dist_mat
            )

            return matches

        return process(keys_)

    def compute(self, original_images, motion_images):
        keys_ = self._extract_key_points(original_images, motion_images)
        matches = self._extract_matches(keys_)

        # FIXME: calculate velocity with motion center

        mask = matches['velocity_norm'] < self.__p_max_velocity
        matches.apply_filter(mask)

        matches['local_motion_center_a'] = matches['local_center_a']
        matches['local_motion_center_b'] = matches['local_center_b']
        matches['global_motion_center_a'] = matches['global_center_a']
        matches['global_motion_center_b'] = matches['global_center_b']

        class ComputationResult(dict):
            velocity: np.ndarray = ...
            velocity_x: np.ndarray = ...
            velocity_y: np.ndarray = ...
            velocity_norm: np.ndarray = ...
            valid: bool = ...

            def __get_accessor(self, suffix):
                class Accessor:
                    local_center: np.ndarray = ...
                    global_local_center: np.ndarray = ...
                    frame: np.ndarray = ...
                    local_motion_center: np.ndarray = ...
                    global_motion_center: np.ndarray = ...
                    match_index: np.ndarray = ...

                    def __init__(self, result):
                        self._result = result

                    def __getattribute__(self, name):
                        value = super().__getattribute__('_result').get(f'{name}_{suffix}')
                        if value is not None:
                            return value
                        return super().__getattribute__(name)

                return Accessor(self)

            def __init__(self, matches, additional_dict):
                super().__init__(matches | additional_dict)
                self.matches = matches
                self.a = self.__get_accessor('a')
                self.b = self.__get_accessor('b')

            def __getattribute__(self, name):
                value = super().__getattribute__('get')(name)
                if value is not None:
                    return value
                return super().__getattribute__(name)

        return ComputationResult(matches, dict(valid=True))


def main():
    from util import motions, originals

    i = 191
    motion_images = motions[i], motions[i + 1]
    original_images = originals[i], originals[i + 1]

    rect = slice(50, 250), slice(100, 300)

    detector = ExtremaFeatureMotionDetector(
        detection_region_rect=rect
    )

    result = detector.compute(original_images, motion_images)

    import matplotlib.pyplot as plt

    fig = plt.figure()

    import matplotlib.animation as animation

    def animate(i):
        ax = fig.gca()
        ax.cla()

        for s, d in zip(result.a.global_motion_center, result.b.global_motion_center):
            ax.arrow(
                s[1],
                s[0],
                d[1] - s[1],
                d[0] - s[0],
                color='red',
                width=1
            )
        ax.imshow(motion_images[i])

    ani = animation.FuncAnimation(fig, animate, interval=500, frames=2, blit=False, save_count=50)

    ani.save('anim.gif')

    from tqdm import tqdm

    matches_tuple = [tuple(x) for x in zip(result.a.match_index, result.b.match_index)]
    n_keys_a = result.matches.keys['count_a']
    n_keys_b = result.matches.keys['count_b']
    fig, axes = plt.subplots(n_keys_a + 2, n_keys_b + 2, figsize=(40, 40))

    for i in tqdm(range(n_keys_a)):
        for j in range(n_keys_b):
            axes[i + 2, j + 2].bar([0], [result.matches.dist_mat[i, j]])
            axes[i + 2, j + 2].set_ylim(0, 1)
            if (i, j) in matches_tuple:
                axes[i + 2, j + 2].scatter([0], [0.5], color='red', s=500)

    for i in range(n_keys_a):
        axes[i + 2, 0].imshow(original_images[0])
        axes[i + 2, 0].scatter(
            result.matches.keys['global_center_a'][i, 1],
            result.matches.keys['global_center_a'][i, 0],
            color='yellow',
            marker='x',
            s=200
        )
        axes[i + 2, 1].imshow(result.matches.keys['frame_a'][i])
    for i in range(n_keys_b):
        axes[0, i + 2].imshow(original_images[0])
        axes[0, i + 2].scatter(
            result.matches.keys['global_center_b'][i, 1],
            result.matches.keys['global_center_b'][i, 0],
            color='yellow',
            marker='x',
            s=200
        )
        axes[1, i + 2].imshow(result.matches.keys['frame_b'][i])
    for ax in axes.flatten():
        ax.axis('off')
    fig.tight_layout()
    fig.savefig('local_max_feature_dist_mat.jpg')
    fig.show()


if __name__ == '__main__':
    main()
