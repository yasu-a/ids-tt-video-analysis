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
            max_movement=20
    ):
        self.__p_detection_region_rect = detection_region_rect
        self.__c_rect = *self.__p_detection_region_rect, slice(None, None)
        self.__p_mean_conv_win_size_factor = mean_conv_win_size_factor
        self.__p_motion_local_max_distance_factor = motion_local_max_distance_factor
        self.__p_motion_local_max_thresh = motion_local_max_thresh
        self.__p_mutual_match_max_cos_distance = mutual_match_max_cos_distance
        self.__p_key_image_size = key_image_size // 2
        self.__p_max_movement = max_movement

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

    def compute(self, original_images, motion_images):
        motion_a, motion_b = motion_images
        original_a, original_b = original_images

        motion_a = self._process_input(motion_a).mean(axis=2)
        motion_b = self._process_input(motion_b).mean(axis=2)
        original_a = self._process_input(original_a)
        original_b = self._process_input(original_b)

        motion_a_mean, motion_b_mean = self._process_mean(motion_a), self._process_mean(motion_b)

        motion_a_local_max = self._local_max(motion_a_mean)
        motion_b_local_max = self._local_max(motion_b_mean)

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

        result = dict(
            valid=False,
            original_a=original_a,
            original_b=original_b,
            motion_a=motion_a,
            motion_b=motion_b,
            key_img_a=key_img_a,
            key_img_b=key_img_b,
            motion_a_local_max=motion_a_local_max,
            motion_b_local_max=motion_b_local_max
        )

        if not (key_img_a and key_img_b):
            return result  # no motion detected

        dist_mat = self._compare_key_batches(key_img_a, key_img_b)
        matches = self._find_mutual_best_match(dist_mat)
        movements = np.linalg.norm(
            motion_b_local_max[matches[:, 1]] - motion_a_local_max[matches[:, 0]],
            axis=1
        )
        mask = movements <= self.__p_max_movement
        matches = matches[mask]

        cx, cy = self.__c_rect[0].start, self.__c_rect[1].start

        src_point = motion_a_local_max[matches[:, 0]]
        dst_point = motion_b_local_max[matches[:, 1]]
        movements = movements[mask]
        angles = np.arctan2(
            (dst_point - src_point)[:, 0], (dst_point - src_point)[:, 1]
        ) * 180 / np.pi

        return result | dict(
            valid=True,
            matches=matches,
            dist_mat=dist_mat,
            src_rect=src_point,
            dst_rect=dst_point,
            src=src_point + [cx, cy],
            dst=dst_point + [cx, cy],
            movements=movements,
            angle=angles
        )


def main():
    from util import motions, originals

    i = 210
    motion_images = motions[i], motions[i + 1]
    original_images = originals[i], originals[i + 1]

    rect = slice(50, 250), slice(100, 300)

    detector = ExtremaFeatureMotionDetector(
        detection_region_rect=rect
    )

    result = detector.compute(original_images, motion_images)
    matches = result['matches']
    key_img_a = result['key_img_a']
    key_img_b = result['key_img_b']
    dist_mat = result['dist_mat']
    motion_a_local_max = result['motion_a_local_max']
    motion_b_local_max = result['motion_b_local_max']
    src, dst = result['src'], result['dst']

    print(matches)

    import matplotlib.pyplot as plt

    fig = plt.figure()

    import matplotlib.animation as animation

    def animate(i):
        ax = fig.gca()
        ax.cla()

        img = motion_images[i]
        for s, d in zip(src, dst):
            ax.arrow(
                s[1],
                s[0],
                d[1] - s[1],
                d[0] - s[0],
                color='red',
                width=1
            )
        ax.imshow(img)

    ani = animation.FuncAnimation(fig, animate, interval=500, frames=2, blit=False, save_count=50)

    ani.save('anim.gif')

    from tqdm import tqdm

    matches_tuple = [tuple(x) for x in matches]
    fig, axes = plt.subplots(len(key_img_a) + 2, len(key_img_b) + 2, figsize=(40, 40))
    for i in tqdm(range(len(key_img_a))):
        for j in range(len(key_img_b)):
            axes[i + 2, j + 2].bar([0], [dist_mat[i, j]])
            axes[i + 2, j + 2].set_ylim(0, 1)
            if (i, j) in matches_tuple:
                axes[i + 2, j + 2].scatter([0], [0.5], color='red', s=500)

    for i in range(len(key_img_a)):
        axes[i + 2, 0].imshow(original_images[0])
        axes[i + 2, 0].scatter(motion_a_local_max[i, 1], motion_a_local_max[i, 0], color='yellow',
                               marker='x', s=200)
        axes[i + 2, 1].imshow(key_img_a[i])
    for i in range(len(key_img_b)):
        axes[0, i + 2].imshow(original_images[0])
        axes[0, i + 2].scatter(motion_b_local_max[i, 1], motion_b_local_max[i, 0], color='yellow',
                               marker='x', s=200)
        axes[1, i + 2].imshow(key_img_b[i])
    for ax in axes.flatten():
        ax.axis('off')
    fig.tight_layout()
    fig.savefig('local_max_feature_dist_mat.jpg')
    fig.show()


if __name__ == '__main__':
    main()
