'''
Lucas-Kanade tracker
====================

Lucas-Kanade sparse optical flow demo. Uses goodFeaturesToTrack
for track initialization and back-tracking for match verification
between frames.

Usage
-----
lk_track.py [<video_source>]


Keys
----
ESC - exit
'''

import os

import cv2
import numpy as np

from config import config

lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

feature_params = dict(maxCorners=500,
                      qualityLevel=0.2,
                      minDistance=7,
                      blockSize=7)


class App:
    TRACK_LEN = 10
    DETECT_INTERVAL = 1

    def __init__(self, video_src):
        self.tracks: list[list[tuple[float, float]]] = []
        self.__cap = cv2.VideoCapture(video_src)
        self.__frame_idx = 0

    def run(self):
        while True:
            ret, img = self.__cap.read()
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_vis = img.copy()

            if len(self.tracks) > 0:
                img_gray_prev, img_gray_cur = self.prev_gray, img_gray
                p0 = np.float32([tr[-1] for tr in self.tracks]).reshape(-1, 1, 2)
                p1, _, _ = cv2.calcOpticalFlowPyrLK(img_gray_prev, img_gray_cur, p0, None,
                                                    **lk_params)
                p0r, _, _ = cv2.calcOpticalFlowPyrLK(img_gray_cur, img_gray_prev, p1, None,
                                                     **lk_params)
                good = abs(p0 - p0r).squeeze().max(-1) <= 0.5

                new_tracks = []
                for tr, (x, y), good_flag in zip(self.tracks, p1.reshape(-1, 2), good):
                    if not good_flag:
                        continue
                    tr.append((x, y))
                    if len(tr) > self.TRACK_LEN:
                        del tr[0]
                    new_tracks.append(tr)
                    # cv2.circle(img_vis, (x, y), 2, (0, 255, 0), -1)
                self.tracks = new_tracks
                # cv2.polylines(img_vis, [np.int32(tr) for tr in self.tracks], False, (0, 255, 0))
                for tr in self.tracks:
                    tr = np.int32(tr)
                    cv2.arrowedLine(img_vis, tr[-1] - (tr[-1] - tr[-2]) * 5, tr[-1], (0, 255, 255),
                                    thickness=3, tipLength=0.2)
                # draw_str(img_vis, (20, 20), 'track count: %d' % len(self.tracks))

            if self.__frame_idx % self.DETECT_INTERVAL == 0:
                mask = np.zeros_like(img_gray)
                mask[:] = 255
                for x, y in [np.int32(tr[-1]) for tr in self.tracks]:
                    cv2.circle(mask, (x, y), 5, 0, -1)
                p = cv2.goodFeaturesToTrack(img_gray, mask=mask, **feature_params)
                if p is not None:
                    for x, y in np.float32(p).reshape(-1, 2):
                        self.tracks.append([(x, y)])

            self.__frame_idx += 1
            # if self.__frame_idx > 60:
            #     break
            self.prev_gray = img_gray
            cv2.imshow('lk_track', img_vis)

            ch = 0xFF & cv2.waitKey(1)
            if ch == 27:
                break


def main():
    video_name = config.default_video_name
    video_src = os.path.join(config.video_location, video_name + '.mp4')
    App(video_src).run()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
