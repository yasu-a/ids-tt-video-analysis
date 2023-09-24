import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage

import util

np.set_printoptions(suppress=True)

from tqdm import tqdm
import skimage.feature

motions, originals, tss = util.load()

RECT = slice(50, 250), slice(100, 300)  # height, width

MEAN_CONV_WIN_SIZE_FACTOR = 32
_mean_conv_win_shape = np.array(motions[0].shape)[:-1] // MEAN_CONV_WIN_SIZE_FACTOR
mean_conv_win = np.ones(_mean_conv_win_shape, dtype=np.float32)
mean_conv_win = mean_conv_win / mean_conv_win.sum()


def process_mean(fr):
    fr_mean = scipy.ndimage.convolve(
        fr,
        weights=mean_conv_win,
        mode='constant',
    )

    return fr_mean


def process_frame_general(fr):
    fr = fr.astype(np.float32) / 256.0
    fr = fr[RECT[0], RECT[1], :]
    return fr


LOCAL_MAX_DISTANCE_FACTOR = 32
LOCAL_MAX_THRESH = 0.03


def local_max(img):
    points = skimage.feature.peak_local_max(
        img,
        min_distance=max(img.shape) // LOCAL_MAX_DISTANCE_FACTOR
    )
    return points[img[tuple(points.T)] > LOCAL_MAX_THRESH]
    # return points[[img[tuple(points.T)].argmax()]]


# fig, axes = plt.subplots(3, 2, figsize=(10, 10))
# axes = axes.flatten()
# for ax, img in zip(axes,
#                    [motion_a, motion_b, original_a, original_b, motion_a_mean, motion_b_mean]):
#     ax.imshow(img)
#
# axes[4].scatter(motion_a_local_max[:, 1], motion_a_local_max[:, 0], color='red', marker='x',
#                 alpha=0.7)
# axes[5].scatter(motion_b_local_max[:, 1], motion_b_local_max[:, 0], color='red', marker='x',
#                 alpha=0.7)
# fig.show()


def compare_images(images_a, images_b):
    def split_3x3(a):
        n = a.shape[0] // 3  # assuming t.shape[0] == t.shape[1]
        m = n * 2
        slices = slice(None, n), slice(n, m), slice(m, None)
        splits = [a[s1, s2] for s1 in slices for s2 in slices]
        return splits

    def extract_feature(img):
        r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
        feature = []
        for t in [r, g, b]:
            for p in split_3x3(t):
                hist, _ = np.histogram(p, bins=16, range=(0, 1))
                feature.append(hist)
        total_feature = np.concatenate(feature)
        return total_feature / total_feature.sum()

    feature_a = list(map(extract_feature, images_a))
    feature_b = list(map(extract_feature, images_b))

    from sklearn.metrics.pairwise import cosine_similarity

    dist_mat = 1 - np.clip(cosine_similarity(feature_a, feature_b), 0, 1)

    return dist_mat


MUTUAL_MATCH_MAX_DISTANCE = 0.3


def find_mutual_best_match(dist_mat):
    best_forward = dist_mat.argmin(axis=1)  # x to y (1st dim to 2nd dim)
    x_index = np.arange(dist_mat.shape[0])
    best_backward = dist_mat.argmin(axis=0)  # y to x
    y_index = np.arange(dist_mat.shape[1])

    mutual_love_from_forward = best_backward[best_forward] == x_index
    x = x_index[mutual_love_from_forward]
    y = y_index[best_forward[mutual_love_from_forward]]

    assert x.size == y.size, (x.shape, y.shape)

    mask = dist_mat[x, y] < MUTUAL_MATCH_MAX_DISTANCE
    x, y = x[mask], y[mask]

    return np.stack([x, y]).T


def main():
    i = 191
    motion_a = process_frame_general(motions[i]).max(axis=2)
    motion_b = process_frame_general(motions[i + 1]).max(axis=2)
    original_a = process_frame_general(originals[i])
    original_b = process_frame_general(originals[i + 1])
    motion_a_mean = process_mean(motion_a)
    motion_b_mean = process_mean(motion_b)

    motion_a_local_max = local_max(motion_a_mean)
    motion_b_local_max = local_max(motion_b_mean)

    KEY_IMAGE_SIZE = 32 // 2

    key_img_a = util.extract_frames_around(
        original_a,
        x=motion_a_local_max[:, 1],
        y=motion_a_local_max[:, 0],
        size=KEY_IMAGE_SIZE
    )
    key_img_b = util.extract_frames_around(
        original_b,
        x=motion_b_local_max[:, 1],
        y=motion_b_local_max[:, 0],
        size=KEY_IMAGE_SIZE
    )

    dm = compare_images(key_img_a, key_img_b)
    matches = [tuple(x) for x in find_mutual_best_match(dm)]
    print(matches)

    fig, axes = plt.subplots(len(key_img_a) + 2, len(key_img_b) + 2, figsize=(40, 40))
    for i in tqdm(range(len(key_img_a))):
        for j in range(len(key_img_b)):
            axes[i + 2, j + 2].bar([0], [dm[i, j]])
            axes[i + 2, j + 2].set_ylim(0, 1)
            if (i, j) in matches:
                axes[i + 2, j + 2].scatter([0], [0.5], color='red', s=500)

    for i in range(len(key_img_a)):
        axes[i + 2, 0].imshow(original_a)
        axes[i + 2, 0].scatter(motion_a_local_max[i, 1], motion_a_local_max[i, 0], color='yellow',
                               marker='x', s=200)
        axes[i + 2, 1].imshow(key_img_a[i])
    for i in range(len(key_img_b)):
        axes[0, i + 2].imshow(original_b)
        axes[0, i + 2].scatter(motion_b_local_max[i, 1], motion_b_local_max[i, 0], color='yellow',
                               marker='x', s=200)
        axes[1, i + 2].imshow(key_img_b[i])
    for ax in axes.flatten():
        ax.axis('off')
    fig.tight_layout()
    fig.show()


if __name__ == '__main__':
    main()
