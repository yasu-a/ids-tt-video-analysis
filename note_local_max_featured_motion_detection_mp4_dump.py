import sys

import matplotlib.pyplot as plt
import numpy as np

from util import motions, originals

np.set_printoptions(suppress=True)

from tqdm import tqdm

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
for i in tqdm(range(200, 400)):
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

ani.save('anim.gif')
