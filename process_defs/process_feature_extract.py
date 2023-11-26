import argparse

import numpy as np

import app_logging
import npstorage_context as snp_context
import process
import storage
import storage.npstorage as snp

logger = app_logging.create_logger(__name__)


class ProcessStageFeatureExtract(process.ProcessStage):
    NAME = 'feature-extract'
    ALIASES = 'fe',

    @classmethod
    def customize_parser(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument('video_names', type=str, nargs='+')
        parser.add_argument('--out-path', type=str, default='./fe-out.csv')

    def __init__(
            self,
            video_names: list[str],
            out_path: str
    ):
        self.__video_names = video_names
        logger.info(f'{video_names=!r}')

        self.__out_path = out_path
        logger.info(f'{out_path=!r}')

    def video_to_csv(self, video_name):
        def motion_matrix_remove_nan_row(motion_matrix):
            return motion_matrix[~np.logical_or(np.isnan(motion_matrix), axis=1), :]

        with storage.create_instance(
                domain='numpy_storage',
                entity=video_name,
                context='primitive_motion',
                mode='r',
        ) as snp_primitive_motion:
            assert isinstance(snp_primitive_motion, snp.NumpyStorage)

            for i in range(0, snp_primitive_motion.count()):
                pmd_entry = snp_primitive_motion.get_entry(i)
                assert isinstance(pmd_entry, snp_context.SNPEntryPrimitiveMotion)

                start_mat, end_mat = pmd_entry.start, pmd_entry.end
                start_mat = motion_matrix_remove_nan_row(start_mat)
                end_mat = motion_matrix_remove_nan_row(end_mat)
                assert len(start_mat) == len(end_mat)

                motion_start = start_mat
                motion_velocity = end_mat - start_mat

                # FIXME: normalize motion coordination by source frame size

    def run(self):
        for video_name in self.__video_names:
            self.video_to_csv(video_name)
