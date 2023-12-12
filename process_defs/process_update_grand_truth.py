import argparse
import os

import app_logging
import process

logger = app_logging.create_logger(__name__)


class ProcessStageUpdateGrandTruth(process.ProcessStage):
    NAME = 'update-grand-truth'
    ALIASES = 'ugt',

    @classmethod
    def customize_parser(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument('-n', '--min-num-sources', type=int, default=3)
        parser.add_argument('-c', '--clear-old', action='store_true')

    def __init__(self, min_num_sources, clear_old):
        self.__min_num_sources = min_num_sources
        self.__clear_old = clear_old

    @classmethod
    def list_video_names(cls):
        root = './label_data/markers'
        for video_name in os.listdir(root):
            dir_path = os.path.join(root, video_name)
            import glob
            n_sources = len(glob.glob(os.path.join(dir_path, '*.json')))
            yield video_name, dir_path, n_sources  # TODO: don't count file, see aggregation result

    @classmethod
    def remove_grand_truth_csvs(cls):
        root = './label_data/grand_truth'
        import shutil
        try:
            shutil.rmtree(root)
        except FileNotFoundError:
            pass

    def run(self):
        if self.__clear_old:
            logger.info('Clearing GT dataframes...')
            self.remove_grand_truth_csvs()
            logger.info('Done')

        logger.info('Updating GT dataframes...')

        from util_ground_truth_generator import GrandTruthGenerator
        from label_manager.frame_label.factory import VideoFrameLabelFactory

        fac = VideoFrameLabelFactory.create_instance()
        gtg = GrandTruthGenerator(fac)

        for video_name, _, n_sources in self.list_video_names():
            if n_sources < self.__min_num_sources:
                logger.warning(f'GT not generated: lack of sources')
                continue
            try:
                gtg.dump_grand_truth_dataframe(video_name)
            except FileNotFoundError as e:
                logger.warning(f'GT not generated: {e}')

        logger.info('Done!')
