import sys

import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage

import util
from util import motions, originals, tss

np.set_printoptions(suppress=True)

from tqdm import tqdm

from util_extrema_feature_motion_detector import ExtremaFeatureMotionDetector

import train_input


def create_rally_mask():
    s, e = train_input_df.start.to_numpy(), train_input_df.end.to_numpy()
    r = np.logical_and(s <= tss[:, None], tss[:, None] <= e).sum(axis=1)
    r = r > 0
    return r.astype(np.uint8)


VIDEO_NAME = '20230205_04_Narumoto_Harimoto'
train_input_df = train_input.load(f'./train/iDSTTVideoAnalysis_{VIDEO_NAME}.csv')
rally_mask = create_rally_mask()

rect = slice(70, 260), slice(180, 255)  # height, width
# height: 奥の選手の頭から手前の選手の足がすっぽり入るように
# width: ネットの部分の卓球台の幅に合うように

w = rect[1].stop - rect[1].start
aw = int(w * 1.0)
rect = slice(rect[0].start, rect[0].stop), slice(rect[1].start - aw, rect[1].stop + aw)

detector = ExtremaFeatureMotionDetector(
    rect
)

i = 193
result = detector.compute(
    original_images=(originals[i], originals[i + 1]),
    motion_images=(motions[i], motions[i + 1])
)

import skimage.feature


# def correct_motion_center(src_original, dst_original):
#     tp = skimage.feature.match_template(
#         dst_original,
#         src_original,
#         pad_input=True,
#         mode='constant'
#     )
#     center = np.array(dst_original.shape)[:-1] // 2
#     match = np.concatenate(np.where(tp == tp.max()))[:-1]
#     correction = match - center  # dst_original centroid - src_original centroid
#     return correction


print(result.matches.n_matches)
fig, axes = plt.subplots(5, result.matches.n_matches, figsize=(6 * result.matches.n_matches, 6 * 4))
for j in range(result.matches.n_matches):
    feature_frs = []
    for i, t in enumerate('ab'):
        fr = result.matches[f'frame_{t}'][j]
        fil = np.array([
            [0, 0.5, 0],
            [0, 0, 0],
            [0, -0.5, 0]
        ])
        grad = np.sqrt(
            np.square(scipy.ndimage.convolve(fr.mean(axis=2), fil, mode='nearest')) \
            + np.square(scipy.ndimage.convolve(fr.mean(axis=2), fil.T, mode='nearest'))
        )
        grad = skimage.filters.rank.mean(grad, np.ones((3, 3)))
        mask = grad > np.percentile(grad, 50)
        grad[~mask] = 0
        # feature_points = skimage.feature.peak_local_max(grad, min_distance=1)
        axes[[0, 3][i], j].imshow(grad)
        # axes[[1, 2][i], j].scatter(
        #     feature_points[:, 1], feature_points[:, 0], marker='x', s=500,
        #     color='red', linewidths=5, alpha=0.7
        # )
        axes[[1, 2][i], j].imshow(fr)
        feature_frs.append(np.where(np.tile(mask[..., None], 3), fr, 0))
    tp = skimage.feature.match_template(feature_frs[1], feature_frs[0], pad_input=True,
                                        mode='constant')
    center = np.array(feature_frs[1].shape)[:-1] // 2
    x, y = np.meshgrid(np.arange(tp.shape[0]), np.arange(tp.shape[1]))
    x, y = x - center[0], y - center[1]
    r = int(tp.shape[0] * 0.5) // 2
    tp[x * x + y * y > r * r] = 0
    cx, cy = a = np.concatenate(np.where(tp == tp.max()))[:-1]
    dx, dy = a - center
    print(result.matches['global_center_b'][j], dx, dy)
    axes[4, j].imshow(tp)
    axes[4, j].arrow(*center, dy, dx, width=0.5, color='red')
    axes[4, j].scatter(
        [cy], [cx], marker='x', s=500,
        color='red', linewidths=5, alpha=0.7
    )
for ax in axes.flatten():
    ax.axis('off')
fig.tight_layout()
fig.show()
plt.close()
