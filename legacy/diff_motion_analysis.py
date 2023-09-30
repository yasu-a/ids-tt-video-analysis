import os
import sys
import time

import numpy as np
import matplotlib.pyplot as plt


def get_latest_path(target):
    dct = {}
    for name in os.listdir('../out'):
        pure_name, _ = os.path.splitext(name)
        video_name, ts = pure_name.rsplit('_', maxsplit=1)
        ts = int(ts)
        if video_name == target:
            dct[ts] = name
    lst = sorted(dct.items(), key=lambda x: x[0])
    target_name = lst[-1][1]
    return os.path.join('../out', target_name)


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
motion_v, motion_h, ts = npz['arr_0'], npz['arr_1'], npz['arr_2']
CLAMP = 4000
motion_v = motion_v[:CLAMP].copy()
motion_h = motion_h[:CLAMP].copy()
ts = ts[:CLAMP].copy()

print('motion_v', motion_v.shape)
print('motion_h', motion_h.shape)
print('ts', ts.shape)

import seaborn as sns

import train_input

train_input_df = train_input.load(f'./train/iDSTTVideoAnalysis_{VIDEO_NAME}.csv')


def create_rally_mask():
    s, e = train_input_df.start.to_numpy(), train_input_df.end.to_numpy()
    r = np.logical_and(s <= ts[:, None], ts[:, None] <= e).sum(axis=1)
    r = r > 0
    return r


rally_mask = create_rally_mask()

print(ts.min(), ts.max())


def ax_plot_heatmap(ax, motion_name):
    motion = globals()[motion_name]
    sns.heatmap(motion.T, ax=ax, cbar=False)
    ax.set_title(f'{VIDEO_NAME} {motion_name}')


def ax_plot_rally_mask(ax):
    sns.heatmap(rally_mask[None, :], ax=ax, cbar=False)


def motion_center(motion):
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


def ax_plot_motion_center(ax, motion_name):
    motion = globals()[motion_name]
    ax.plot(-motion_center(motion), lw=0.5)
    ax.set_title(f'{motion_name}')


import scipy.ndimage


def smooth(a, sigma=None):
    return scipy.ndimage.gaussian_filter1d(a, sigma or 10)


def quarters(motion):
    quarter_size = motion.shape[1] // 3
    dct_motion = dict(
        full=motion,
        left=motion[:, :quarter_size],
        middle=motion[:, quarter_size:-quarter_size],
        right=motion[:, -quarter_size:]
    )
    return dct_motion


def map_kinds(motion, f):
    return {k: f(motion_) for k, motion_ in quarters(motion).items()}


def ax_plot_std(ax, motion_name):
    motion = globals()[motion_name]
    for label, v in map_kinds(motion, lambda m: np.std(m, axis=1)).items():
        ax.plot(v, lw=0.5, label=label)
    ax.set_title(f'{motion_name} std')
    ax.legend()
    ax.grid()


def ax_plot_mean(ax, motion_name):
    motion = globals()[motion_name]
    for label, v in map_kinds(motion, lambda m: np.mean(m, axis=1)).items():
        ax.plot(v, lw=0.5, label=label)
    ax.set_title(f'{motion_name} mean')
    ax.legend()
    ax.grid()
    ax.set_yscale('log')


def ax_edit_ticklabel(ax):
    ax.set_xticklabels(ts[ax.get_xticks().astype(int)].round(1))


plot_lst = [
    [ax_plot_rally_mask],
    [ax_plot_heatmap, 'motion_v'],
    [ax_plot_motion_center, 'motion_v'],
    [ax_plot_std, 'motion_v'],
    [ax_plot_mean, 'motion_v'],
    [ax_plot_rally_mask],
    [ax_plot_heatmap, 'motion_h'],
    [ax_plot_motion_center, 'motion_h'],
    [ax_plot_std, 'motion_h'],
    [ax_plot_mean, 'motion_h'],
    [ax_plot_rally_mask],
]

fig, axes = plt.subplots(
    len(plot_lst), 1,
    figsize=(100, 15),
    gridspec_kw={'height_ratios': [1 if f == ax_plot_rally_mask else 5 for f, *_ in plot_lst]},
    sharex=True
)

for i in tqdm(range(len(plot_lst))):
    ax = axes[i]
    f, *args = plot_lst[i]
    args = ax, *args
    f(*args)

plt.tight_layout()
plt.savefig('out.png')
plt.show()

sys.exit(0)
