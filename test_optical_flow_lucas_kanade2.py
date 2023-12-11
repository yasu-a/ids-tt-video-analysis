import datetime
import functools
import os
from dataclasses import dataclass
from pprint import pformat
from typing import NamedTuple, Any, Literal, Optional, Iterable

import cv2
import numpy as np
from tqdm import tqdm

import app_logging
import train_input
from config import config
from train_input import RectNormalized, RectActualScaled

logger = app_logging.create_logger('__name__')


@dataclass(frozen=True)
class LKVideoExtractorParameter:
    rect: RectNormalized
    lk_params: dict[str, Any]
    feature_params: dict[str, Any]
    original_resize_scale: float
    max_track_length_seconds: float
    min_producible_track_length: float
    min_valid_track_length_seconds: float
    detect_interval_frames: int
    # ↓ normalized by rect and timestamp [unit width(height)/second]
    min_velocity_full_normalized: float
    imshow: bool = False
    video_dump_path: str = None


def _default_param(video_name):
    return LKVideoExtractorParameter(
        rect=train_input.frame_rects.normalized(video_name),
        lk_params=dict(winSize=(15, 15),
                       maxLevel=2,
                       criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)),
        feature_params=dict(maxCorners=256,
                            qualityLevel=0.1,
                            minDistance=5,
                            blockSize=5),
        original_resize_scale=0.4,
        max_track_length_seconds=0.3,
        min_producible_track_length=0.1,
        min_valid_track_length_seconds=0.05,
        detect_interval_frames=1,
        min_velocity_full_normalized=0.05,
        imshow=False,
        video_dump_path='./out.mp4'
    )


@dataclass(frozen=True)
class Frame:
    original: cv2.typing.MatLike
    timestamp: float
    frame_index: int


class VideoMeta(NamedTuple):
    frame_count: int
    frame_rate: float
    frame_width: int
    frame_height: int

    __CV2_VC_PROP_MAPPING = dict(
        frame_count=cv2.CAP_PROP_FRAME_COUNT,
        frame_rate=cv2.CAP_PROP_FPS,
        frame_width=cv2.CAP_PROP_FRAME_WIDTH,
        frame_height=cv2.CAP_PROP_FRAME_HEIGHT
    )

    @classmethod
    def from_video_capture(cls, cap: cv2.VideoCapture) -> 'VideoMeta':
        field_type = cls.__annotations__
        kwargs = {
            field_name: field_type[field_name](cap.get(cv2_prop))
            for field_name, cv2_prop in cls.__CV2_VC_PROP_MAPPING.items()
        }

        return cls(**kwargs)


class FrameProducer:
    def __init__(self, video_path: str):
        self._video_path = video_path

    @classmethod
    def from_video_name(cls, video_name: str):
        return cls(
            video_path=os.path.join(
                config.video_location,
                video_name + '.mp4'
            )
        )

    class FrameIterator:
        def __init__(self, fp: 'FrameProducer'):
            self.__fp = fp
            logger.info(f'{self.__fp._video_path=}')
            self.__cap = cv2.VideoCapture(self.__fp._video_path)

        def meta(self):
            return VideoMeta.from_video_capture(self.__cap)

        def __next__(self) -> 'Frame':
            frame_index = int(self.__cap.get(cv2.CAP_PROP_POS_FRAMES))

            ret, img = self.__cap.read()
            if not ret:
                raise StopIteration()

            timestamp = float(self.__cap.get(cv2.CAP_PROP_POS_MSEC)) / 1e+3

            return Frame(
                original=img,
                timestamp=timestamp,
                frame_index=frame_index
            )

        def __iter__(self):
            return self

        def __enter__(self) -> 'FrameProducer.FrameIterator':
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            self.__cap.release()
            return False

        def __len__(self):
            return self.meta().frame_count

    def __iter__(self):
        return self.FrameIterator(self)

    @property
    def meta(self):
        with iter(self) as it:
            return it.meta()


class Track:
    def __init__(self, p: LKVideoExtractorParameter):
        self.__p = p
        self.__xy: list[tuple[int, int]] = []
        self.__timestamps: list[float] = []

    @property
    def timedelta(self):
        return max(self.__timestamps) - min(self.__timestamps)

    def append(self, x: int, y: int, timestamp: float):
        self.__xy.append((x, y))
        self.__timestamps.append(timestamp)
        while self.timedelta > self.__p.max_track_length_seconds:
            del self.__timestamps[0]
            del self.__xy[0]

    def keypoint(self) -> tuple[int, int]:
        return self.__xy[-1]

    @property
    def valid(self):
        return self.timedelta >= self.__p.min_valid_track_length_seconds

    @property
    def producible(self):
        return self.timedelta >= self.__p.min_producible_track_length

    def velocity(
            self,
            *,
            reliability: Literal['valid', 'producible'],
            invariance: Literal['fps', 'rect', 'both'] = None,
            rect: RectActualScaled = None
    ) -> Optional[tuple[int | float, int | float]]:
        if reliability == 'valid':
            if not self.valid:
                return None
        elif reliability == 'producible':
            if not self.producible:
                return None
        else:
            assert False, reliability

        if invariance == 'both':
            invariance = 'fps,rect'
        invariance = (invariance or '').split(',')

        points = np.array(self.__xy)
        delta_points = np.diff(points, axis=0)
        return_type = int

        if 'fps' in invariance:
            timestamps = np.array(self.__timestamps)
            delta_timestamps = np.diff(timestamps)
            delta_points /= delta_timestamps[:, None]
            return_type = float

        if 'rect' in invariance:
            assert rect is not None
            delta_points /= (rect.size.x, rect.size.y)
            return_type = float

        x, y = delta_points.mean(axis=0)
        x, y = return_type(x), return_type(y)

        return x, y


class Tracks:
    def __init__(self, p: LKVideoExtractorParameter):
        self.__p = p
        self.__tracks: list[Track] = []

    def __len__(self):
        return len(self.__tracks)

    def __getitem__(self, i) -> Track:
        return self.__tracks[i]

    def __new_track(self):
        return Track(p=self.__p)

    def add_new_keypoints(self, points, timestamp):
        for x, y in points:
            track = self.__new_track()
            track.append(x, y, timestamp)
            self.__tracks.append(track)

    def update_tracks(self, new_points, mask, timestamp: float):
        assert len(self) == len(new_points) == len(mask), \
            (len(self), len(new_points), len(mask))

        for i in reversed(range(len(self))):
            if mask[i]:
                self.__tracks[i].append(new_points[i][0], new_points[i][1], timestamp)
            else:
                del self.__tracks[i]

    def keypoint(self, *, i32=False) -> np.ndarray:  # like list[tuple[float, float]]
        assert self  # tracks exist

        def it():
            for track in self.__tracks:
                yield track.keypoint()

        arr = np.int32 if i32 else np.float32
        return arr(list(it()))

    def velocity(self, **kwargs) -> np.ndarray:  # like list[tuple[float, float]]
        assert self  # tracks exist

        def it():
            for track in self.__tracks:
                v = track.velocity(**kwargs)
                if v is None:
                    yield [np.nan, np.nan]
                else:
                    yield v

        return np.float32(list(it()))

    def velocity_norm(self, **kwargs) -> np.ndarray:  # like list[float]
        return np.linalg.norm(self.velocity(**kwargs), axis=1)

    def producible(self):
        def it():
            for track in self.__tracks:
                yield track.producible

        return np.bool_(list(it()))

    def remove_tracks_by_mask(self, mask):
        for i in reversed(range(len(self))):
            if mask[i]:
                del self.__tracks[i]


@dataclass(frozen=True)
class LKDetectorFrame(Frame):
    target: cv2.typing.MatLike
    target_gray: cv2.typing.MatLike


LKDetectorResultType = tuple[LKDetectorFrame, Tracks]
LKResultIteratorType = Iterable[LKDetectorResultType]


# [SEE](http://labs.eecs.tottori-u.ac.jp/sd/Member/oyamada/OpenCV/html/py_tutorials/py_video
# /py_lucas_kanade/py_lucas_kanade.html)
class LKMotionComputer:
    def __init__(self, p: LKVideoExtractorParameter, fp: FrameProducer):
        self.__p = p
        self.__fp = fp
        self.__tracks: Tracks = Tracks(p)

    @functools.cached_property
    def __actual_scaled_rect(self):
        meta = self.__fp.meta
        return self.__p.rect.to_actual_scaled(
            width=meta.frame_width,
            height=meta.frame_height
        )

    def _process_lk(self, fr_prev: LKDetectorFrame, fr: LKDetectorFrame):
        # トラックが存在した場合
        if self.__tracks:
            # `min_velocity_full_normalized`よりも小さい速さの動きのトラックは...
            full_normalized_velocity = self.__tracks.velocity_norm(
                reliability='valid',
                invariance='fps',
                rect=self.__actual_scaled_rect
            )
            np.nan_to_num(
                full_normalized_velocity,
                copy=False,
                nan=np.inf
            )
            mask_too_slow = full_normalized_velocity < self.__p.min_velocity_full_normalized
            # 除外する
            self.__tracks.remove_tracks_by_mask(mask_too_slow)

        # 比較対象が読み込まれていてトラックが存在する場合
        if fr_prev is not None and self.__tracks:
            # 存在するトラックの fr_prev -> fr へのトラック点を探す
            p0 = self.__tracks.keypoint()
            # noinspection PyTypeChecker
            p1, _, _ = cv2.calcOpticalFlowPyrLK(
                fr_prev.target_gray,
                fr.target_gray,
                p0.astype(np.float32).reshape(-1, 1, 2),
                None,
                **self.__p.lk_params
            )

            # fr_prev -> fr へのトラック点から fr -> fr_prev へのトラック点を逆にたどる
            # noinspection PyTypeChecker
            p0r, _, _ = cv2.calcOpticalFlowPyrLK(
                fr.target_gray,
                fr_prev.target_gray,
                p1,
                None,
                **self.__p.lk_params
            )

            # fr_prev -> fr -> fr_prev へのトラック点がもとに戻ってきたかどうかで`good`マスクを生成
            good = np.abs(p0 - p0r.reshape(-1, 2)).max(-1) <= 1

            # `good`な点だけトラックに追加する
            self.__tracks.update_tracks(p1.reshape(-1, 2), good, fr.timestamp)

        # `detect_interval_frames`の度に
        if fr.frame_index % self.__p.detect_interval_frames == 0:
            # キーポイント検出領域用のマスクをつくる：
            #  トラック済みの点の半径`cv2.circle(radius=?)`は検出しない
            mask = np.zeros_like(fr.target_gray)
            mask[:] = 255
            if self.__tracks:
                for x, y in self.__tracks.keypoint(i32=True):
                    # noinspection PyTypeChecker
                    cv2.circle(mask, (x, y), 5, 0, -1)

            # キーポイントを抽出する
            p = cv2.goodFeaturesToTrack(fr.target_gray, mask=mask, **self.__p.feature_params)
            # 検出できたら
            if p is not None:
                p = np.float32(p).reshape(-1, 2)  # list of xy
                # 新たなトラックとして追加する
                self.__tracks.add_new_keypoints(p, fr.timestamp)

    def _process_frame_image(self, fr: Frame) -> LKDetectorFrame:
        original_resized = cv2.resize(
            fr.original,
            None,
            fx=self.__p.original_resize_scale,
            fy=self.__p.original_resize_scale
        )

        target = fr.original[self.__actual_scaled_rect.index3d]
        fr = LKDetectorFrame(
            original=original_resized,
            frame_index=fr.frame_index,
            timestamp=fr.timestamp,
            target=target,
            target_gray=cv2.cvtColor(target, cv2.COLOR_BGR2GRAY),
        )

        return fr

    def _iter_results(self) -> LKResultIteratorType:
        logger.info(pformat(self.__p))

        fr_prev = None

        with iter(self.__fp) as frame_iter:
            bar = tqdm(frame_iter)
            for _fr in bar:
                fr = self._process_frame_image(_fr)

                self._process_lk(fr_prev, fr)
                fr_prev = fr

                bar.set_description(
                    f'{datetime.timedelta(seconds=fr.timestamp)} {len(self.__tracks)}'
                )

                yield fr, self.__tracks

    @classmethod
    def _draw_motion_vectors(cls, img: cv2.typing.MatLike, tracks: Tracks):
        # キーポイント，速度，使用に適しているかのフラグを取得する
        start_arr = tracks.keypoint()
        velocity_arr = tracks.velocity(
            reliability='producible',
            invariance='fps'
        )
        flag_arr = tracks.producible()

        # 矢印を描画する
        # noinspection PyTypeChecker
        it_zip = zip(start_arr, velocity_arr, flag_arr)
        for start, velocity, producible in it_zip:
            if not producible:
                continue
            start = start.astype(np.int32)
            velocity = velocity.astype(np.int32)
            cv2.arrowedLine(
                img,
                start,
                start + velocity,
                (0, 255, 255),
                thickness=1,
                tipLength=0.2
            )

    def _iter_wrapper_imshow(self, it: LKResultIteratorType) -> LKResultIteratorType:
        for fr, tr in it:
            if self.__p.imshow:
                # 画像を暗くしてモーションベクトルを見やすくする
                img = fr.target.copy()
                cv2.LUT(
                    img,
                    (np.arange(256) * 0.8).astype(np.uint8)
                )

                # TODO: may draws 2 times
                self._draw_motion_vectors(img, self.__tracks)

                # imshow
                cv2.imshow('lk_track', img)

                ch = 0xFF & cv2.waitKey(1)
                if ch == 27:
                    break

            yield fr, tr

    def _iter_wrapper_video_dump(self, it: LKResultIteratorType) -> LKResultIteratorType:
        import async_writer

        with async_writer.AsyncVideoFrameWriter(
                path=self.__p.video_dump_path,
                fps=self.__fp.meta.frame_rate / self.__p.detect_interval_frames
        ) as vw:
            for fr, tr in it:
                if self.__p.imshow:
                    # 画像を暗くしてモーションベクトルを見やすくする
                    img = fr.target.copy()
                    cv2.LUT(
                        img,
                        (np.arange(256) * 0.8).astype(np.uint8)
                    )

                    # TODO: may draws 2 times
                    self._draw_motion_vectors(img, self.__tracks)

                    # 動画を書き込む
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    vw.write(img)

                yield fr, tr

    def iter_results(self) -> LKResultIteratorType:
        it = self._iter_results()

        if self.__p.imshow:
            it = self._iter_wrapper_imshow(it)

        if self.__p.video_dump_path:
            it = self._iter_wrapper_video_dump(it)

        return it


class LKMotionDetector:
    def __init__(self, parameter: LKVideoExtractorParameter):
        self.__p = parameter

    def computer(self, frame_producer: FrameProducer):
        return LKMotionComputer(self.__p, frame_producer)


def main():
    video_name = config.default_video_name
    detector = LKMotionDetector(
        parameter=_default_param(video_name)
    )
    it = detector.computer(
        frame_producer=FrameProducer.from_video_name(video_name),
    ).iter_results()
    for item in it:
        _ = item
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
