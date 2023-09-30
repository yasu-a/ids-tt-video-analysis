import sys

import matplotlib.pyplot as plt
import numpy as np

from legacy.util import motions, originals, tss

np.set_printoptions(suppress=True)

from tqdm import tqdm

from util_extrema_feature_motion_detector import ExtremaFeatureMotionDetector

import train_input

VIDEO_NAME = '20230205_04_Narumoto_Harimoto'
train_input_df = train_input.load(f'./train/iDSTTVideoAnalysis_{VIDEO_NAME}.csv')

print(train_input_df)


def create_rally_mask():
    s, e = train_input_df.start.to_numpy(), train_input_df.end.to_numpy()
    r = np.logical_and(s <= tss[:, None], tss[:, None] <= e).sum(axis=1)
    r = r > 0
    return r.astype(np.uint8)


rally_mask = create_rally_mask()

print(rally_mask)

rect = slice(70, 260), slice(180, 255)  # height, width
# height: 奥の選手の頭から手前の選手の足がすっぽり入るように
# width: ネットの部分の卓球台の幅に合うように

w = rect[1].stop - rect[1].start
aw = int(w * 1.0)
rect = slice(rect[0].start, rect[0].stop), slice(rect[1].start - aw, rect[1].stop + aw)

detector = ExtremaFeatureMotionDetector(
    rect
)


def split_quarter(index_array, src_shape):
    n = src_shape[0] // 3
    m = n * 2

    y = index_array[:, 0]

    idx_vec = np.arange(len(index_array))

    return (
        idx_vec[y < n],
        idx_vec[(n <= y) & (y < m)],
        idx_vec[m < y]
    )


S, E = 200, 1000
print(tss[S], tss[E])

stack = dict(
    mv_std=[[], [], []],
    mv_mean=[[], [], []],
)
video_dump = dict(
    image=[],
    src=[],
    dst=[]
)

for i in tqdm(range(S, E)):
    motion_images = motions[i], motions[i + 1]
    original_images = originals[i], originals[i + 1]
    result = detector.compute(original_images, motion_images)
    video_dump['image'].append(originals[i])

    if not result['valid']:
        src, dst = [], []
        for i in range(3):
            stack['mv_std'][i].append(0)
            stack['mv_mean'][i].append(0)
    else:
        src, dst = result.a.global_motion_center, result.b.global_motion_center
        for i, idx in enumerate(split_quarter(src, result['motion_a'].shape)):
            if idx.size == 0:
                stack['mv_std'][i].append(0)
                stack['mv_mean'][i].append(0)
            else:
                stack['mv_std'][i].append(np.std(result.velocity_y))
                stack['mv_mean'][i].append(np.mean(result.velocity_y))

    video_dump['src'].append(src)
    video_dump['dst'].append(dst)

# chart
for k in stack.keys():
    stack[k] = np.array(stack[k])

import seaborn as sns

fig, axes = plt.subplots(4, 1, sharex=True, figsize=(60, 10))
sns.heatmap(rally_mask[None, S:E], ax=axes[0], cbar=False)
sns.heatmap(rally_mask[None, S:E], ax=axes[-1], cbar=False)

for i, name in enumerate(['mv_std', 'mv_mean']):
    i = i + 1
    axes[i].set_title(name)
    for j in range(3):
        lst = stack[name][j]
        axes[i].plot(lst, label=f'{j}')
    axes[i].legend()


def ax_edit_ticklabel(ax):
    ax.set_xticklabels(tss[S:E][ax.get_xticks().astype(int)].round(1), rotation=90)


ax_edit_ticklabel(axes[-1])

plt.tight_layout()

plt.show()
plt.close()

# video
fig = plt.figure()

import matplotlib.animation as animation

bar = tqdm(total=len(video_dump['image']))


def animate(j):
    ax = fig.gca()
    ax.cla()

    for s, d in zip(video_dump['src'][j], video_dump['dst'][j]):
        ax.arrow(
            s[1],
            s[0],
            d[1] - s[1],
            d[0] - s[0],
            color='white',
            width=1
        )
        ax.arrow(
            s[1],
            s[0],
            d[1] - s[1],
            d[0] - s[0],
            color='red',
            width=0.5
        )
    ax.imshow(video_dump['image'][j])

    bar.update()


ani = animation.FuncAnimation(
    fig,
    animate,
    interval=100,
    frames=len(video_dump['image']),
    blit=False,
    save_count=50
)

ani.save('out.mp4')

sys.exit(0)
