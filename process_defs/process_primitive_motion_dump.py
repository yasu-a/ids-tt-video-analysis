import argparse
from typing import Optional

import numpy as np
from tqdm import tqdm

import app_logging
import npstorage_context as snp_context
import process
import storage
import storage.npstorage as snp
import train_input
from primitive_motion_detector._detector import PMDetector
from train_input import RectActualScaled

logger = app_logging.create_logger(__name__)


def _pad_motion_matrix(mat, max_n):
    if mat is None:
        return np.full((max_n, 2), np.nan)
    mat = mat.astype(np.float32)
    n_pad = max_n - mat.shape[0]
    pad = np.full((n_pad, 2), np.nan)
    mat = np.concatenate([mat, pad], axis=0)
    assert len(mat) == max_n, mat.shape
    return mat


class ProcessStagePrimitiveMotionDump(process.ProcessStage):
    NAME = 'primitive-motion-dump'
    ALIASES = 'pmd',
    ENABLED = False

    @classmethod
    def customize_parser(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument('video_name', type=str)
        parser.add_argument('-n', '--max-points-per-frame', '--max-points', type=int, default=64)

    def __init__(
            self,
            video_name: str,
            max_points_per_frame: int
    ):
        self.__video_name = video_name
        logger.info(f'{video_name=}')

        self.__max_points_per_frame = max_points_per_frame
        logger.info(f'{max_points_per_frame=}')

    def run(self):
        with storage.create_instance(
                domain='numpy_storage',
                entity=self.__video_name,
                context='frames',
                mode='r',
        ) as snp_video_frame:
            assert isinstance(snp_video_frame, snp.NumpyStorage)

            with storage.create_instance(
                    domain='numpy_storage',
                    entity=self.__video_name,
                    context='primitive_motion',
                    mode='w',
                    n_entries=snp_video_frame.count()
            ) as snp_primitive_motion:
                assert isinstance(snp_primitive_motion, snp.NumpyStorage)

                detector: Optional[PMDetector] = None
                rect_actual_scaled: Optional[RectActualScaled] = None

                for i in tqdm(range(snp_video_frame.count() - 1)):
                    snp_vf_entry_current = snp_video_frame.get_entry(i)
                    snp_vf_entry_next = snp_video_frame.get_entry(i + 1)
                    assert isinstance(snp_vf_entry_current, snp_context.SNPEntryVideoFrame)
                    assert isinstance(snp_vf_entry_next, snp_context.SNPEntryVideoFrame)

                    if detector is None:
                        source_frame_height = snp_vf_entry_current.motion.shape[0]
                        source_frame_width = snp_vf_entry_current.motion.shape[1]

                        rect_actual_scaled = train_input.frame_rects.actual_scaled(
                            video_name=self.__video_name,
                            height=source_frame_height,
                            width=source_frame_width
                        )

                        detector = PMDetector(
                            detection_region_rect=rect_actual_scaled
                        )

                        logger.info('Detector created:')
                        logger.info(f' * {detector=}')
                        logger.info(f' * {rect_actual_scaled=}')

                    assert detector is not None and rect_actual_scaled is not None

                    originals = (
                        snp_vf_entry_current.original,
                        snp_vf_entry_next.original
                    )
                    motions = (
                        snp_vf_entry_current.motion,
                        snp_vf_entry_next.motion
                    )
                    timestamp = snp_vf_entry_current.timestamp
                    fi = snp_vf_entry_current.fi

                    detection_result = detector.compute(
                        original_images=originals,
                        motion_images=motions
                    )

                    if detection_result['valid']:
                        start_mat = rect_actual_scaled.normalize_points_inside_based_on_corner(
                            detection_result.a.local_motion_center
                        )
                        end_mat = rect_actual_scaled.normalize_points_inside_based_on_corner(
                            detection_result.b.local_motion_center
                        )

                        start_mat = _pad_motion_matrix(
                            start_mat,
                            max_n=self.__max_points_per_frame
                        )
                        end_mat = _pad_motion_matrix(
                            end_mat,
                            max_n=self.__max_points_per_frame
                        )
                    else:
                        start_mat = _pad_motion_matrix(
                            None,
                            max_n=self.__max_points_per_frame
                        )
                        end_mat = _pad_motion_matrix(
                            None,
                            max_n=self.__max_points_per_frame
                        )

                    snp_primitive_motion[i] = snp_context.SNPEntryPrimitiveMotion(
                        start=start_mat,
                        end=end_mat,
                        timestamp=timestamp,
                        fi=fi
                    )
