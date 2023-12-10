import os
from typing import NamedTuple

import cv2
import numpy as np

import train_input
from config import config

lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

feature_params = dict(maxCorners=256,
                      qualityLevel=0.1,
                      minDistance=5,
                      blockSize=5)

ORIGINAL_RESIZE_SCALE = 0.4


class Frame(NamedTuple):
    original: cv2.typing.MatLike
    target: cv2.typing.MatLike
    target_gray: cv2.typing.MatLike
    timestamp: float
    frame_index: int


MAX_TRACK_LENGTH = 0.3
MIN_PRODUCE_TRACK_LENGTH = 0.1
MIN_TRACK_LENGTH = 0.05


class Track:
    def __init__(self):
        self.__xy: list[tuple[int, int]] = []
        self.__timestamps: list[float] = []

    @property
    def timedelta(self):
        return max(self.__timestamps) - min(self.__timestamps)

    def append(self, x: int, y: int, timestamp: float):
        self.__xy.append((x, y))
        self.__timestamps.append(timestamp)
        while self.timedelta > MAX_TRACK_LENGTH:
            del self.__xy[0]
            del self.__timestamps[0]

    @property
    def last_point(self):  # -> like tuple[int, int]:
        return np.int32(self.__xy[-1])

    def motion_vector(self, homogeneous=False):  # -> like tuple[float, float] | None:
        if self.timedelta < MIN_TRACK_LENGTH:
            if homogeneous:
                return np.array([np.nan, np.nan])
            else:
                return None
        xys = np.array(self.__xy)
        ts = np.array(self.__timestamps)
        delta_fps_invariant = np.diff(xys, axis=0) / np.diff(ts)[:, None]
        return delta_fps_invariant.mean(axis=0)  # normalized by timestamp

    def result(self):
        if self.timedelta < MIN_PRODUCE_TRACK_LENGTH:
            return None
        return self.last_point, self.motion_vector()


class Tracks:
    def __init__(self):
        self.__tracks: list[Track] = []

    def __len__(self):
        return len(self.__tracks)

    def __getitem__(self, i) -> Track:
        return self.__tracks[i]

    def add_new_points(self, points, timestamp):
        for x, y in points:
            track = Track()
            track.append(x, y, timestamp)
            self.__tracks.append(track)

    def update(self, new_points, mask, timestamp):
        assert len(self) == len(new_points) == len(mask), (len(self), len(new_points), len(mask))

        for i in reversed(range(len(self))):
            if mask[i]:
                self.__tracks[i].append(new_points[i][0], new_points[i][1], timestamp)
            else:
                del self.__tracks[i]

    def get_keypoints(self) -> np.ndarray:  # list[tuple[int, int]]
        assert self
        return np.int32([track.last_point for track in self.__tracks])

    def velocity(self, rect: train_input.RectActualScaled) -> np.ndarray:  # like list[float]
        assert self
        vs = np.stack([tr.motion_vector(homogeneous=True) for tr in self.__tracks])
        vs_normalized = vs / (rect.size.x, rect.size.y)
        return np.linalg.norm(vs_normalized, axis=1)

    def remove(self, mask):
        for i in reversed(range(len(self))):
            if mask[i]:
                del self.__tracks[i]

    def iter_results(self):
        for tr in self.__tracks:
            result = tr.result()
            if result is None:
                continue
            yield result  # self.last_point, self.motion_vector()


class App:
    DETECT_INTERVAL = 1
    MIN_VELOCITY_FULL_NORMALIZED = 0.05

    def __init__(self, video_name):
        self._tracks: Tracks = Tracks()
        self.__cap = cv2.VideoCapture(
            os.path.join(config.video_location, video_name + '.mp4')
        )
        w = int(self.__cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.__cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.__rect: train_input.RectActualScaled \
            = train_input.frame_rects.actual_scaled(video_name, width=w, height=h)

    def iter_frames(self):
        while True:
            frame_index = int(self.__cap.get(cv2.CAP_PROP_POS_FRAMES))

            ret, img = self.__cap.read()
            if not ret:
                return

            timestamp = float(self.__cap.get(cv2.CAP_PROP_POS_MSEC)) / 1e+3

            img_resized = cv2.resize(
                img,
                None,
                fx=ORIGINAL_RESIZE_SCALE,
                fy=ORIGINAL_RESIZE_SCALE
            )
            img_target = img[self.__rect.index3d]

            yield Frame(
                original=img_resized,
                target=img_target,
                target_gray=cv2.cvtColor(img_target, cv2.COLOR_BGR2GRAY),
                timestamp=timestamp,
                frame_index=frame_index
            )

    def run(self):
        fr_prev = None

        for fr in self.iter_frames():
            if self._tracks:
                mask_too_slow \
                    = self._tracks.velocity(self.__rect) < self.MIN_VELOCITY_FULL_NORMALIZED
                self._tracks.remove(mask_too_slow)

            img_vis = fr.target.copy()

            if fr_prev is not None and self._tracks:
                p0 = self._tracks.get_keypoints()
                # noinspection PyTypeChecker
                p1, _, _ = cv2.calcOpticalFlowPyrLK(
                    fr_prev.target_gray,
                    fr.target_gray,
                    p0.astype(np.float32).reshape(-1, 1, 2),
                    None,
                    **lk_params
                )
                # noinspection PyTypeChecker
                p0r, _, _ = cv2.calcOpticalFlowPyrLK(
                    fr.target_gray,
                    fr_prev.target_gray,
                    p1,
                    None,
                    **lk_params
                )
                good = np.abs(p0 - p0r.reshape(-1, 2)).max(-1) <= 1

                self._tracks.update(p1.reshape(-1, 2), good, fr.timestamp)
                # cv2.circle(img_vis, (x, y), 2, (0, 255, 0), -1)
                # cv2.polylines(img_vis, [np.int32(tr) for tr in self.tracks], False, (0, 255, 0))

                for start, velocity in self._tracks.iter_results():
                    end = (start + velocity).round(0).astype(int)
                    cv2.arrowedLine(img_vis, start, end, (0, 255, 255),
                                    thickness=1, tipLength=0.2)
                # draw_str(img_vis, (20, 20), 'track count: %d' % len(self.tracks))

            if fr.frame_index % self.DETECT_INTERVAL == 0:
                # generate detection mask
                mask = np.zeros_like(fr.target_gray)
                mask[:] = 255
                if self._tracks:
                    for x, y in self._tracks.get_keypoints():
                        # noinspection PyTypeChecker
                        cv2.circle(mask, (x, y), 5, 0, -1)

                # extract keypoints
                p = cv2.goodFeaturesToTrack(fr.target_gray, mask=mask, **feature_params)
                if p is not None:
                    p = np.float32(p).reshape(-1, 2)  # list of xy
                    self._tracks.add_new_points(p, fr.timestamp)

            fr_prev = fr
            cv2.imshow('lk_track', img_vis)

            ch = 0xFF & cv2.waitKey(1)
            if ch == 27:
                break


def main():
    video_name = config.default_video_name
    App(video_name).run()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
