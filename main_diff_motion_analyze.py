import sys

import cv2
import matplotlib.pyplot as plt
import scipy.ndimage

import util
import json
import os

import numpy as np

np.set_printoptions(suppress=True)

from main_diff_generator_bug_fix import MEMMAP_PATH
from tqdm import tqdm
import skimage.feature


def load():
    motions = np.memmap(
        os.path.join(MEMMAP_PATH, 'motions.map'),
        mode='r',
        dtype=np.uint8
    )
    originals = np.memmap(
        os.path.join(MEMMAP_PATH, 'originals.map'),
        mode='r',
        dtype=np.uint8
    )
    tss = np.memmap(
        os.path.join(MEMMAP_PATH, 'tss.map'),
        mode='r',
        dtype=np.float32
    )

    with open(os.path.join(MEMMAP_PATH, 'shape.json'), 'r') as f:
        shape_json = json.load(f)

    motions = motions.reshape(shape_json['motions'])
    originals = originals.reshape(shape_json['originals'])
    tss = tss.reshape(shape_json['tss'])

    return motions, originals, tss


motions, originals, tss = load()

print(f'{motions.shape=} {originals.shape=} {tss.shape=}')

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


i = 191
motion_a = process_frame_general(motions[i]).max(axis=2)
motion_b = process_frame_general(motions[i + 1]).max(axis=2)
original_a = process_frame_general(originals[i])
original_b = process_frame_general(originals[i + 1])
motion_a_mean = process_mean(motion_a)
motion_b_mean = process_mean(motion_b)

motion_a_local_max = local_max(motion_a_mean)
motion_b_local_max = local_max(motion_b_mean)

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


# def process_image(img):
#     r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
#     # gs = img.mean(axis=2)
#     targets = [r, g, b]
#     total_diffs = []
#     for t in targets:
#         diff_1 = np.abs(t[1:, 1:] - t[:-1, 1:])
#         diff_2 = np.abs(t[1:, 1:] - t[1:, :-1])
#         total_diff = (diff_1 + diff_2) / 2
#         total_diffs.append(total_diff[..., None])
#     return np.concatenate(total_diffs, axis=2)


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

# fig, axes = plt.subplots(3, 1, sharex=True)
# axes[0].hist(motion_a_mean[:, :50].flatten(), bins=100, label='r', alpha=0.4, color='green')
# axes[1].hist(motion_a_mean[:, 50:150].flatten(), bins=100, label='c', alpha=0.4, color='red')
# axes[2].hist(motion_a_mean[:, 150:].flatten(), bins=100, label='l', alpha=0.4, color='blue')
# axes[0].set_yscale('log')
# axes[1].set_yscale('log')
# axes[2].set_yscale('log')
# fig.show()

sys.exit(0)


def split1d(a_src, n):
    a = a_src.copy()
    if a.size % n != 0:
        pad = np.zeros(n - a.size % n)
        pad[:] = a_src[-1]
        a = np.concatenate([a, pad])
    assert a.size % n == 0, (n, a.shape)
    return np.array(np.split(a, n))


SPLIT_COUNT = 32

import scipy.signal


def _extract_motion_single_diff(a, b, axis):
    a_am = a.argmax(axis=axis)
    b_am = b.argmax(axis=axis)
    velocity = (b_am - a_am) / a.shape[axis]
    strength = np.sqrt(a.max(axis=axis) * b.max(axis=axis))
    amount = velocity * strength
    amount = scipy.signal.medfilt(amount, kernel_size=5)
    return velocity, strength, amount


def extract_motion(a, b, c, axis):
    v, s, a = _extract_motion_single_diff(a, b, axis)
    vv, ss, aa = _extract_motion_single_diff(b, c, axis)
    relevance = np.sqrt(np.sqrt(np.square(v - vv) * np.square(s - ss)))
    return (v + vv) / 2, (s + ss) / 2, (a + aa) / 2 * relevance


def create_motion_feature(a, b, c):
    # plt.figure()
    a = process_frame(a)
    # plt.imshow(a)
    b = process_frame(b)
    c = process_frame(c)
    v_0, s_0, a_0 = extract_motion(a, b, c, axis=0)  # .size == width  -> horizontal motion
    v_1, s_1, a_1 = extract_motion(a, b, c, axis=1)  # .size == height -> vertical motion
    return dict(
        frame_first=a,
        frame_second=b,
        horizontal_velocity=v_0,
        horizontal_strength=s_0,
        horizontal_amount=a_0,
        vertical_velocity=v_1,
        vertical_strength=s_1,
        vertical_amount=a_1,
        width=v_0.size,
        height=v_1.size
    )


def draw_motion_feature_plot(mf, ax):
    im = np.concatenate([
        mf['frame_first'],
        mf['frame_second']
    ], axis=1)
    im = np.concatenate([
        im,
        np.concatenate([
            mf['frame_second'],
            np.zeros_like(mf['frame_second'])
        ], axis=1)
    ], axis=0)
    ax.imshow(im)
    for i, v, s, amount in zip(np.arange(mf['height']), mf['vertical_velocity'],
                               mf['vertical_strength'], mf['vertical_amount']):
        plt.arrow(-25, mf['height'] + i, amount * mf['width'] * 20, 0, width=1, color=(0, 1, 0),
                  alpha=0.3)
    for i, v, s, amount in zip(np.arange(mf['width']), mf['horizontal_velocity'],
                               mf['horizontal_strength'], mf['horizontal_amount']):
        plt.arrow(mf['width'] + i, -25, 0, amount * mf['height'] * 20, width=1, color=(0, 1, 0),
                  alpha=0.3)
    ax.set_xlim(mf['width'] * 2, -100)
    ax.set_ylim(mf['height'] * 2, -100)


def extract_motion(a, b, max_velocity):
    N = 16

    feature_points_a = util.local_max_2d(a)
    feature_parts_a = util.extract_frames_around(
        a, feature_points_a[:, 1], feature_points_a[:, 0], N
    )

    feature_points_b = util.local_max_2d(b)
    feature_parts_b = util.extract_frames_around(
        b, feature_points_b[:, 1], feature_points_b[:, 0], N
    )

    if feature_points_a.size > 0 and feature_points_b.size > 0:
        dist = []
        for fa in feature_parts_a:
            dist.append([])
            for fb in feature_parts_b:
                dist[-1].append(np.sum(np.square(fa - fb)))

        dist = np.array(dist)
        dist = np.sqrt(dist)

        forward = dist.argmin(axis=1)
        backward = dist.argmin(axis=0)

        motion_feature_target_a = np.arange(len(forward))[
            backward[forward] == np.arange(len(forward))]
        motion_feature_target_b = forward[motion_feature_target_a]

        motion_velocity = feature_points_b[motion_feature_target_b] - feature_points_a[
            motion_feature_target_a]

        mask = np.sqrt(np.sum(np.square(motion_velocity), axis=1)) <= max_velocity

        a_points = feature_points_a[motion_feature_target_a][mask]
        b_points = feature_points_b[motion_feature_target_b][mask]
        velocity = motion_velocity[mask]
    else:
        a_points = np.array([])
        b_points = np.array([])
        velocity = np.array([])

    return dict(
        a_points=a_points,
        b_points=b_points,
        velocity=velocity
    )


a = motions[191]
a = process_frame(a)
b = motions[192]
b = process_frame(b)

import io
from PIL import Image


def fig2img(fig):
    """Convert a Matplotlib figure to a PIL Image and return it"""

    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    return img


from tqdm import tqdm

ims = []
for i in tqdm(range(150, 700)):
    a, b = process_frame(motions[i]), process_frame(motions[i + 1])
    fig = plt.figure(figsize=(5, 7))
    motion = extract_motion(a, b, max_velocity=32)
    fig.gca().imshow(b)
    for src, v in zip(motion['a_points'], motion['velocity']):
        fig.gca().arrow(
            src[1],
            src[0],
            v[1],
            v[0],
            color='red',
            width=1
        )
    img = fig2img(fig)
    ims.append(img)
    plt.close()

ims[0].save("out.gif", format="GIF", append_images=ims,
            save_all=True, duration=166, loop=0)
