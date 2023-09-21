import os
import sys
import time

import numpy as np
import matplotlib.pyplot as plt


def get_latest_path(target):
    dct = {}
    for name in os.listdir('./out'):
        pure_name, _ = os.path.splitext(name)
        video_name, ts = pure_name.rsplit('_', maxsplit=1)
        ts = int(ts)
        if video_name == target:
            dct[ts] = name
    lst = sorted(dct.items(), key=lambda x: x[0])
    target_name = lst[-1][1]
    return os.path.join('./out', target_name)


VIDEO_NAME = '20230205_04_Narumoto_Harimoto'

SRC_NPY_PATH = get_latest_path(VIDEO_NAME)
print(SRC_NPY_PATH)

while True:
    try:
        npz = np.load(SRC_NPY_PATH)
        print(npz)
    except EOFError:
        continue
    break
    time.sleep(0.1)
motion, ts = npz['arr_0'], npz['arr_1']
CLAMP = 4000
motion = motion[:CLAMP].copy()
ts = ts[:CLAMP].copy()

print(motion.shape, ts.shape)

import seaborn as sns

import train_input

train_input_df = train_input.load(f'./train/iDSTTVideoAnalysis_{VIDEO_NAME}.csv')


def rally_mask():
    s, e = train_input_df.start.to_numpy(), train_input_df.end.to_numpy()
    r = np.logical_and(s <= ts[:, None], ts[:, None] <= e).sum(axis=1)
    r = r > 0
    return r


print(ts.min(), ts.max())

fig, axes = plt.subplots(
    6, 1,
    figsize=(100, 15),
    gridspec_kw={'height_ratios': [1, 5, 5, 5, 5, 1]},
    sharex=True
)

sns.heatmap(motion.T, ax=axes[1], cbar=False)
axes[1].set_title(VIDEO_NAME)

m = rally_mask()
sns.heatmap(m[None, :], ax=axes[0], cbar=False)
sns.heatmap(m[None, :], ax=axes[-1], cbar=False)


def motion_center():
    N, H = motion.shape[0], motion.shape[1]
    index_vec = np.arange(H)
    center = []
    for i in range(N):
        a = motion[i]
        rank_percent = np.argsort(a) / H
        mask = (0.9 < rank_percent) & (rank_percent < 0.95)
        c = index_vec[mask].mean()
        center.append(c)
    center = np.array(center).round(0).astype(int)
    return center


axes[2].plot(-motion_center(), lw=0.5)

import scipy.ndimage


def smooth(a, sigma=None):
    return scipy.ndimage.gaussian_filter1d(a, sigma or 10)


quarter_size = motion.shape[1] // 3
dct_motion = dict(
    full=motion,
    left=motion[:, :quarter_size],
    middle=motion[:, quarter_size:-quarter_size],
    right=motion[:, -quarter_size:]
)


def map_kinds(f):
    return {k: f(motion_) for k, motion_ in dct_motion.items()}


for label, v in map_kinds(lambda m: np.std(m, axis=1)).items():
    axes[3].plot(v, lw=0.5, label=label)
axes[3].set_title('std')
axes[3].legend()
axes[3].grid()

for label, v in map_kinds(lambda m: np.mean(m, axis=1)).items():
    axes[4].plot(v, lw=0.5, label=label)
axes[4].set_title('mean')
axes[4].legend()
axes[4].grid()
axes[4].set_yscale('log')

axes[-1].set_xticklabels(ts[axes[-1].get_xticks().astype(int)].round(1))

plt.show()

sys.exit(0)