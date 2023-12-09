import collections
import hashlib
import multiprocessing as mp
import queue
import threading
from concurrent.futures import ProcessPoolExecutor
from typing import Optional

import cv2
import numpy as np
import skimage.util
from tqdm import tqdm

import app_logging
import async_writer
import npstorage_context as snp_context
import storage
import storage.npstorage as snp
import train_input
from primitive_motion_detector import *

logger = app_logging.create_logger(__name__)


class GlobalContext:
    BLOCK_TIMEOUT = 0.2

    def __thread_hash(self) -> int:
        hash_source = threading.get_native_id(), threading.get_ident()
        hash_source = str(hash_source).encode('utf-8')
        return int.from_bytes(
            hashlib.md5(hash_source).digest(),
            byteorder='big'
        )

    def __init__(self):
        self.__stop = False
        self.__source_thread_hash = self.__thread_hash()

        self.__q_frames = mp.Queue[snp_context.SNPEntryVideoFrame](maxsize=256)
        self.__q_results = mp.Queue[PMDetectorResult](maxsize=256)

    def __allow_only_main_thread(self):
        assert self.__thread_hash() == self.__source_thread_hash

    def put_frame(self, frame: snp_context.SNPEntryVideoFrame):
        self.__allow_only_main_thread()
        self.__q_frames.put(frame)

    def pop_frame(self) -> Optional[snp_context.SNPEntryVideoFrame]:
        try:
            return self.__q_frames.get(block=True, timeout=self.BLOCK_TIMEOUT)
        except queue.Empty:
            return None

    def put_result(self, result: PMDetectorResult):
        self.__q_results.put(result)

    def pop_result(self) -> Optional[PMDetectorResult]:
        self.__allow_only_main_thread()
        try:
            return self.__q_results.get(block=True, timeout=self.BLOCK_TIMEOUT)
        except queue.Empty:
            return None

    def send_stop(self):
        self.__allow_only_main_thread()
        self.__stop = True

    def resume(self):
        return not self.__stop


# def worker_video_read(ctx: GlobalContext, video_name, start, stop):
#     while ctx.resume():
#         with storage.create_instance(
#                 domain='numpy_storage',
#                 entity=video_name,
#                 context='frames',
#                 mode='r',
#         ) as snp_video_frame:
#             assert isinstance(snp_video_frame, snp.NumpyStorage)
#             for i in range(start, stop):

def create_source(rect: train_input.RectNormalized, snp_video_frame: snp.NumpyStorage, i: int):
    snp_entry_target = snp_video_frame.get_entry(i)
    snp_entry_next = snp_video_frame.get_entry(i + 1)
    assert isinstance(snp_entry_target, snp_context.SNPEntryVideoFrame)
    assert isinstance(snp_entry_next, snp_context.SNPEntryVideoFrame)

    source = PMDetectorSource(
        target_frame=PMDetectorSourceTimeSeriesEntry(
            original_image=snp_entry_target.original,
            diff_image=snp_entry_target.motion,
            timestamp=float(snp_entry_target.timestamp)
        ),
        next_frame=PMDetectorSourceTimeSeriesEntry(
            original_image=snp_entry_next.original,
            diff_image=snp_entry_next.motion,
            timestamp=float(snp_entry_next.timestamp)
        ),
        detection_rect_normalized=rect
    )

    return source


def mp_main():
    video_name = '20230205_04_Narumoto_Harimoto'
    start, stop = 200, 400

    with storage.create_instance(
            domain='numpy_storage',
            entity=video_name,
            context='frames',
            mode='r',
    ) as snp_video_frame:
        assert isinstance(snp_video_frame, snp.NumpyStorage)

        start = start or 0
        stop = stop or snp_video_frame.count()

        tsd = np.diff(snp_video_frame.get_array('timestamp'))
        fps = 1 / tsd[tsd.mean() - tsd.std() <= tsd].mean()

        with async_writer.AsyncVideoFrameWriter(
                path='out.mp4',
                fps=fps
        ) as vw:
            n_workers = max(1, mp.cpu_count() - 1)
            logger.info('{n_workers=}')
            pool = ProcessPoolExecutor(max_workers=n_workers)

            futures = collections.deque()
            indexes = collections.deque(range(start, stop))
            completed_index = start

            rect = train_input.frame_rects.normalized(video_name)

            detector: Optional[PMDetector] = PMDetector(
                PMDetectorParameter(enable_motion_correction=True)
            )

            bar = tqdm(range(start, stop))
            with pool:
                while completed_index < stop:
                    if len(futures) <= 128 and indexes:
                        i = indexes.popleft()

                        source = create_source(
                            rect=rect,
                            snp_video_frame=snp_video_frame,
                            i=i
                        )

                        future = pool.submit(
                            detector.compute,
                            source=source
                        )
                        futures.append(future)

                    if futures:
                        if futures[0].done():
                            result = futures.popleft().result()

                            xs = result.local_centroid

                            img = skimage.util.img_as_ubyte(
                                result.original_images_clipped[0].copy())
                            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                            VIDEO_RESIZE_RATIO = 3
                            img = cv2.resize(img, dsize=None, fx=VIDEO_RESIZE_RATIO,
                                             fy=VIDEO_RESIZE_RATIO)
                            xs = xs * VIDEO_RESIZE_RATIO
                            for p1, p2 in zip(xs[0], xs[1]):
                                cv2.arrowedLine(
                                    img,
                                    p1[::-1],
                                    p2[::-1],
                                    (63, 63, 0),
                                    thickness=3 * VIDEO_RESIZE_RATIO,
                                    tipLength=0.1
                                )
                                cv2.arrowedLine(
                                    img,
                                    p1[::-1],
                                    p2[::-1],
                                    (255, 255, 0),
                                    thickness=1 * VIDEO_RESIZE_RATIO,
                                    tipLength=0.1
                                )
                            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            vw.write(img)

                            completed_index += 1
                            bar.update(1)

                    bar.set_description(f'{len(indexes)=}, {len(futures)=}')


if __name__ == '__main__':
    def main():
        video_name = '20230205_04_Narumoto_Harimoto'
        start, stop = 200, 400

        rect = train_input.frame_rects.normalized(video_name)

        detector: Optional[PMDetector] = PMDetector(
            PMDetectorParameter(enable_motion_correction=True)
        )

        with storage.create_instance(
                domain='numpy_storage',
                entity=video_name,
                context='frames',
                mode='r',
        ) as snp_video_frame:
            assert isinstance(snp_video_frame, snp.NumpyStorage)

            start = start or 0
            stop = stop or snp_video_frame.count()

            tsd = np.diff(snp_video_frame.get_array('timestamp'))
            fps = 1 / tsd[tsd.mean() - tsd.std() <= tsd].mean()

            with async_writer.AsyncVideoFrameWriter(
                    path='out.mp4',
                    fps=fps
            ) as vw:
                for i in tqdm(range(start, stop)):
                    source = create_source(
                        rect=rect,
                        snp_video_frame=snp_video_frame,
                        i=i
                    )
                    result = detector.compute(source=source)

                    xs = result.local_centroid

                    img = skimage.util.img_as_ubyte(result.original_images_clipped[0].copy())
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    for p1, p2 in zip(xs[0], xs[1]):
                        cv2.arrowedLine(
                            img,
                            p1[::-1],
                            p2[::-1],
                            (255, 255, 0),
                        )
                    # img = cv2.resize(img, None, fx=5, fy=5, interpolation=cv2.INTER_CUBIC)
                    # cv2.imshow('win', img)
                    # cv2.waitKey(20)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    vw.write(img)


    mp_main()
