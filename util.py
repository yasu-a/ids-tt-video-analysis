import json
import os

import numpy as np
from skimage.feature import peak_local_max

from main_diff_generator_bug_fix import MEMMAP_PATH

np.set_printoptions(suppress=True)


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

    print(f'{motions.shape=} {originals.shape=} {tss.shape=}')

    return motions, originals, tss


motions, originals, tss = load()


def slice_frame(fr, rect):
    return fr[rect]


def local_max_2d(a):
    THRESH = 0.05
    points = peak_local_max(a)
    return points[a[points[:, 1], points[:, 0]] > THRESH]


def extract_frames_around(a, x, y, size):
    if a.ndim == 3:
        a = np.pad(a, ((size, size), (size, size), (0, 0)), constant_values=0)
    else:
        a = np.pad(a, ((size, size), (size, size)), constant_values=0)
    wins = []
    for xx, yy in zip(x, y):
        cx, cy = xx + size, yy + size
        win = a[cy - size:cy + size, cx - size:cx + size]
        wins.append(win)
    # fig, axes = plt.subplots(1, len(x))
    # for ax, w in zip(axes, wins):
    #     ax.imshow(w)
    # fig.show()
    if wins:
        return np.stack(wins)
    else:
        return None
