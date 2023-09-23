import json
import os

import numpy as np

np.set_printoptions(suppress=True)

from main_diff_generator_bug_fix import MEMMAP_PATH

frames, tss = np.array([]), np.array([])
while True:
    try:
        frames = np.memmap(
            os.path.join(MEMMAP_PATH, 'frames.map'),
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

        frames = frames.reshape(shape_json['frames'])
        tss = tss.reshape(shape_json['tss'])

        print(f'{frames.shape=} {tss.shape=}')
    except OSError:
        continue
    else:
        break

import matplotlib.pyplot as plt
import scipy.ndimage


def split1d(a_src, n):
    a = a_src.copy()
    if a.size % n != 0:
        pad = np.zeros(n - a.size % n)
        pad[:] = a_src[-1]
        a = np.concatenate([a, pad])
    assert a.size % n == 0, (n, a.shape)
    return np.array(np.split(a, n))


RECT = slice(50, 250), slice(150, 300)  # height, width


def slice_frame(fr):
    return fr[RECT]


MEAN_CONV_WIN_SIZE_FACTOR = 32
_mean_conv_win_shape = np.array(frames[0].shape)[:-1] // MEAN_CONV_WIN_SIZE_FACTOR
mean_conv_win = np.ones(_mean_conv_win_shape, dtype=np.float32)
mean_conv_win = mean_conv_win / mean_conv_win.sum()

SPLIT_COUNT = 32

import scipy.signal


def extract_motion(a, b, axis):
    a_am = a.argmax(axis=axis)
    b_am = b.argmax(axis=axis)
    velocity = (b_am - a_am) / a.shape[axis]
    strength = np.sqrt(a.max(axis=axis) * b.max(axis=axis))
    amount = velocity * strength
    amount = scipy.signal.medfilt(amount, kernel_size=5)
    return velocity, strength, amount


def process_frame(fr):
    fr = fr.astype(np.float32) / 256.0

    fr = fr.mean(axis=2)
    fr = slice_frame(fr)

    # plt.figure()
    # plt.imshow(fr)
    # plt.show()

    fr_mean = scipy.ndimage.convolve(
        fr,
        weights=mean_conv_win,
        mode='constant',
    )

    return fr_mean


def create_motion_feature(a, b):
    # plt.figure()
    a = process_frame(a)
    # plt.imshow(a)
    b = process_frame(b)
    v_0, s_0, a_0 = extract_motion(a, b, axis=0)  # .size == width  -> horizontal motion
    v_1, s_1, a_1 = extract_motion(a, b, axis=1)  # .size == height -> vertical motion
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
    # for i, v, s, amount in zip(np.arange(v_1.size), v_1, s_1, a_1):
    #     plt.arrow(v_0.size / 2, i, amount * a.shape[1] * 3, 0, width=1, color=(0, 1, 0), alpha=0.3)
    # plt.show()


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
        plt.arrow(-25, mf['height'] + i, amount * mf['width'] * 3, 0, width=1, color=(0, 1, 0),
                  alpha=0.3)
    for i, v, s, amount in zip(np.arange(mf['width']), mf['horizontal_velocity'],
                               mf['horizontal_strength'], mf['horizontal_amount']):
        plt.arrow(mf['width'] + i, -25, 0, amount * mf['height'] * 3, width=1, color=(0, 1, 0),
                  alpha=0.3)
    ax.set_xlim(mf['width'] * 2, -100)
    ax.set_ylim(mf['height'] * 2, -100)


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
for i in tqdm(range(frames.shape[0] - 1)):
    fig = plt.figure(figsize=(3, 4))
    mf = create_motion_feature(frames[i], frames[i + 1])
    draw_motion_feature_plot(mf, fig.add_subplot(111))
    img = fig2img(fig)
    ims.append(img)
    plt.close()

    if (i + 1) % 64 == 0:
        ims[0].save("out.gif", format="GIF", append_images=ims,
                    save_all=True, duration=166, loop=0)
