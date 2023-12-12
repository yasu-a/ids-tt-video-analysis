import functools
import os.path
from pprint import pformat

import cv2
import numpy as np
import pandas as pd

import app_logging
import npstorage_context as snp_context
import storage
from config import config
from label_manager.frame_label.factory import VideoFrameLabelFactory
from label_manager.frame_label.sample_set import FrameAggregationResult

snp_context.just_run_registration()

storage.context.forbid_writing = True

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.expand_frame_repr', False)

logger = app_logging.create_logger(__name__)


class GrandTruthGenerator:
    def __init__(self, fac: VideoFrameLabelFactory):
        self.__fac = fac

    ORDERED_LABELS = ['Stay', 'Play', 'Ready']
    INTERVAL_LABELS = ORDERED_LABELS[-1], *ORDERED_LABELS

    VIDEO_FRAME_MARGIN = 100  # 最初と最後あたりのフレームは処理上の不具合が多いので取り除く

    @classmethod
    @functools.cache
    def video_frame_count(cls, video_name):
        # TODO: replace with util
        path = os.path.join(
            config.video_location,
            video_name + '.mp4'
        )
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise FileNotFoundError('cap not opened', path)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        frame_count = int(frame_count)
        return frame_count

    @classmethod
    def frame_range(cls, video_name):
        frame_count = cls.video_frame_count(video_name)
        end = frame_count - 1

        return cls.VIDEO_FRAME_MARGIN, end - cls.VIDEO_FRAME_MARGIN

    def create_grand_truth_dataframe(self, video_name):
        logger.info(f'Creating grand truth dataframe for {video_name!r}')

        label_sample_set = self.__fac[video_name]
        logger.debug(f'label_sample_set={pformat(list(label_sample_set))}')

        agg: FrameAggregationResult = label_sample_set.aggregate_full()

        assert set(self.ORDERED_LABELS) == set(label_sample_set.frame_label_name_list), \
            (self.ORDERED_LABELS, label_sample_set.frame_label_name_list)

        lst = list(
            agg.extract_ordered_label_groups(
                label_order=[
                    label_sample_set.frame_label_name_list.index(label_name)
                    for label_name in self.ORDERED_LABELS
                ],
                predicate=lambda entry: entry.reliability > 0.5
            )
        )

        f_start, f_end = self.frame_range(video_name)

        for i in range(len(lst)):
            start = max(
                f_start,
                0 if lst[i].prev_frame is None else lst[i].prev_frame.fi_center
            )
            end = min(
                f_end,
                f_end if lst[i].next_frame is None else lst[i].next_frame.fi_center
            )
            labeled = [
                lst[i].frames[j].fi_center
                for j, _ in enumerate(self.ORDERED_LABELS)
            ]
            start = (start + labeled[0]) // 2
            end = (labeled[-1] + end) // 2
            lst[i] = [start, *labeled, end]

        mat = np.array(lst)  # 2d mat like list of [start, *labels, end]

        logger.debug(mat)

        assert np.all(np.diff(mat.flatten()) >= 0), 'values are not ordered'

        def find_label(fi):
            for mat_row in mat:
                if not (mat_row[0] <= fi < mat_row[-1]):
                    continue
                for j in range(len(mat_row) - 1):
                    if mat_row[j] <= fi < mat_row[j + 1]:
                        return self.INTERVAL_LABELS[j]
            return None

        rows = []
        for i in range(self.video_frame_count(video_name)):
            label = (find_label(i) or 'invalid').lower()
            rows.append({label: True})

        df = pd.DataFrame(rows).fillna(False).astype(bool)

        return df

    def dump_grand_truth_dataframe(self, video_name):
        df = self.create_grand_truth_dataframe(video_name)
        path = os.path.join('label_data/grand_truth', video_name + '.csv')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_csv(path)


if __name__ == '__main__':
    def main():
        video_name_lst = [
            '20230205_04_Narumoto_Harimoto',
            '20230219_03_Narumoto_Ito',
            '20230225_02_Matsushima_Ando'
        ]

        for video_name in video_name_lst:
            gtg = GrandTruthGenerator(VideoFrameLabelFactory.create_instance())
            gtg.dump_grand_truth_dataframe(video_name)


    main()
