import numpy as np
from skimage.feature import peak_local_max

np.set_printoptions(suppress=True)


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
