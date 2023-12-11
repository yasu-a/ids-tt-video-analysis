import argparse

import cv2
import numpy as np
from tqdm import tqdm

import app_logging
import async_writer
import npstorage_context as snp_context
import process
import storage
import storage.npstorage as snp
import train_input

logger = app_logging.create_logger(__name__)


class ProcessStagePrimitiveMotionVisualize(process.ProcessStage):
    NAME = 'primitive-motion-visualize'
    ALIASES = 'pmv',
    ENABLED = False

    @classmethod
    def customize_parser(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument('video_name', type=str)
        parser.add_argument('--start', type=int, default=None)
        parser.add_argument('--stop', type=int, default=None)
        parser.add_argument('--out-path', type=str, default='./vpm-out.mp4')

    def __init__(
            self,
            video_name: str,
            start: int,
            stop: int,
            out_path: str
    ):
        self.__video_name = video_name
        logger.info(f'{video_name=!r}')

        with storage.create_instance(
                domain='numpy_storage',
                entity=self.__video_name,
                context='primitive_motion',
                mode='r',
        ) as snp_primitive_motion:
            assert isinstance(snp_primitive_motion, snp.NumpyStorage)
            count = snp_primitive_motion.count()

        start = start or 0
        stop = stop or count - 1
        self.__start, self.__stop = start, stop
        logger.info(f'{start=}, {stop=}')

        self.__out_path = out_path
        logger.info(f'{out_path=!r}')

    def run(self):
        with storage.create_instance(
                domain='numpy_storage',
                entity=self.__video_name,
                context='frames',
                mode='r',
        ) as snp_video_frames:
            assert isinstance(snp_video_frames, snp.NumpyStorage)

            with storage.create_instance(
                    domain='numpy_storage',
                    entity=self.__video_name,
                    context='primitive_motion',
                    mode='r',
            ) as snp_primitive_motion:
                assert isinstance(snp_primitive_motion, snp.NumpyStorage)

                timestamps = snp_primitive_motion.get_array('timestamp')
                fps = 1 / np.nanmean(np.diff(timestamps))
                logger.info(f'{fps=}')
                assert fps > 0, fps

                with async_writer.AsyncVideoFrameWriter(
                        self.__out_path,
                        fps=fps
                ) as vw:
                    for i in tqdm(range(self.__start, self.__stop)):
                        pmd_entry = snp_primitive_motion.get_entry(i)
                        vfd_entry = snp_video_frames.get_entry(i)
                        assert isinstance(vfd_entry, snp_context.SNPEntryVideoFrame)
                        assert isinstance(pmd_entry, snp_context.SNPEntryPrimitiveMotion)

                        src_frame_height = vfd_entry.motion.shape[0]
                        src_frame_width = vfd_entry.motion.shape[1]

                        assert pmd_entry.fi == vfd_entry.fi, (pmd_entry.fi, vfd_entry.fi)

                        image = vfd_entry.original.copy()
                        image = cv2.resize(image, None, fx=2, fy=2)
                        for m_start, m_end in zip(pmd_entry.start, pmd_entry.end):
                            if np.any(np.isnan(m_start)):
                                break

                            rect = train_input.load_rect(
                                video_name=self.__video_name,
                                height=src_frame_height,
                                width=src_frame_width
                            )
                            rect_y, rect_x = rect
                            rect_height = rect_y.stop - rect_y.start
                            rect_width = rect_x.stop - rect_x.start
                            local_motion_normalize_factor \
                                = 1.0 / np.array([rect_width, rect_height])
                            rect_offset = np.array([rect_x.start, rect_y.start])

                            y1, x1 = (
                                             m_start / local_motion_normalize_factor + rect_offset
                                     ).astype(int) * 2
                            y2, x2 = (
                                             m_end / local_motion_normalize_factor + rect_offset
                                     ).astype(int) * 2
                            # print((x1, y1), (x2, y2))
                            cv2.arrowedLine(image, (x1, y1), (x2, y2), (0, 255, 255), 2)

                        vw.write(image)
