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


SRC_NPY_PATH = get_latest_path('20230205_04_Narumoto_Harimoto')
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

print(motion.shape, ts.shape)

import seaborn as sns

plt.figure(figsize=(25, 5))
sns.heatmap(motion[::5].T)
plt.show()

import scipy.ndimage


def smooth(a, sigma=None):
    return scipy.ndimage.gaussian_filter1d(a, sigma or 10)


quarter_size = motion.shape[1] // 4
motion_quarter_left, motion_quarter_right = motion[:, :quarter_size], motion[:, -quarter_size:]

full_stds = np.std(motion, axis=1)
left_stds = np.std(motion_quarter_left, axis=1)
right_stds = np.std(motion_quarter_right, axis=1)

plt.figure()
plt.plot(ts, full_stds, label='full')
plt.plot(ts, left_stds, label='left')
plt.plot(ts, right_stds, label='right')
plt.title('std')
plt.legend()
plt.show()

sys.exit(0)

peaks = np.percentile(motion, 95, axis=1)
peaks = np.argmin(np.square(peaks[:, None] - motion), axis=1)
peaks = smooth(peaks)
plt.figure()
plt.plot(ts, peaks)
plt.title('95% max')
plt.show()

means = np.percentile(motion, 40, axis=1)
means = smooth(means)
plt.figure()
plt.plot(ts, means)
plt.title('mean')
plt.show()
#
# for i, t in enumerate(ts):
#     frame_motion = motion[i]
#     print(frame_motion)
