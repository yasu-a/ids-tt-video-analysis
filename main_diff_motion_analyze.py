import sys

import cv2
import matplotlib.pyplot as plt
import scipy.ndimage

import util
import json
import os

from util import motions, originals, tss
import numpy as np

np.set_printoptions(suppress=True)

from tqdm import tqdm
import skimage.feature

from util_extrema_feature_motion_detector import ExtremaFeatureMotionDetector

RECT_TEST = False

rect = slice(70, 245), slice(180, 255)  # height, width
# height: 奥の選手の頭から手前の選手の足がすっぽり入るように
# width: ネットの部分の卓球台の幅に合うように

if RECT_TEST:
    plt.figure()
    plt.imshow(originals[110])
    plt.hlines(rect[0].start, rect[1].start, rect[1].stop, colors='white')
    plt.hlines(rect[0].stop, rect[1].start, rect[1].stop, colors='white')
    plt.vlines(rect[1].start, rect[0].start, rect[0].stop, colors='white')
    plt.vlines(rect[1].stop, rect[0].start, rect[0].stop, colors='white')
    plt.show()

w = rect[1].stop - rect[1].start
aw = int(w * 1.0)
rect = slice(rect[0].start, rect[0].stop), slice(rect[1].start - aw, rect[1].stop + aw)

if RECT_TEST:
    plt.figure()
    plt.imshow(originals[110])
    plt.hlines(rect[0].start, rect[1].start, rect[1].stop, colors='white')
    plt.hlines(rect[0].stop, rect[1].start, rect[1].stop, colors='white')
    plt.vlines(rect[1].start, rect[0].start, rect[0].stop, colors='white')
    plt.vlines(rect[1].stop, rect[0].start, rect[0].stop, colors='white')
    plt.show()

    sys.exit(0)

detector = ExtremaFeatureMotionDetector(
    rect
)

frames = []
sources = []
destinations = []
for i in tqdm(range(8000)):
    motion_images = motions[i], motions[i + 1]
    original_images = originals[i], originals[i + 1]
    result = detector.compute(original_images, motion_images)
    if result is None:
        src, dst = [], []
    else:
        src, dst = result['src'], result['dst']
    frames.append(motions[i])
    sources.append(src)
    destinations.append(dst)

fig = plt.figure()

import matplotlib.animation as animation


def animate(j):
    ax = fig.gca()
    ax.cla()

    for s, d in zip(sources[j], destinations[j]):
        ax.arrow(
            s[1],
            s[0],
            d[1] - s[1],
            d[0] - s[0],
            color='red',
            width=1
        )
    ax.imshow(frames[j])


ani = animation.FuncAnimation(fig, animate, interval=100, frames=len(frames), blit=False,
                              save_count=50)

ani.save('anim.mp4')

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
