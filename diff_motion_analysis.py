import time

import numpy as np
import matplotlib.pyplot as plt

while True:
    try:
        npz = np.load('diff_out.npz')
    except EOFError:
        continue
    break
    time.sleep(0.1)
motion, ts = npz['arr_0'], npz['arr_1']

print(motion.shape, ts.shape)

import seaborn as sns

sns.heatmap(motion[::5].T)
plt.show()

import scipy.ndimage


def smooth(a, sigma=None):
    return scipy.ndimage.gaussian_filter1d(a, sigma or 10)


stds = np.std(motion, axis=1)
stds = smooth(stds)
plt.figure()
plt.plot(ts, stds)
plt.title('std')
plt.show()

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
