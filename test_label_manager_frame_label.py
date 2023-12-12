import functools
import os.path
from pprint import pformat
from pprint import pprint

import cv2
import numpy as np
import pandas as pd

import app_logging
import npstorage_context as snp_context
import storage
import storage.npstorage as snp
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
        frame_count = cv2.VideoCapture(
            os.path.join(
                config.video_location,
                video_name + '.mp4'
            )
        ).get(cv2.CAP_PROP_FRAME_COUNT)
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
        logger.info(f'label_sample_set={pformat(list(label_sample_set))}')

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

        logger.info(mat)

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
        df.to_csv(path)


def main():
    video_name_lst = [
        '20230205_04_Narumoto_Harimoto',
        '20230219_03_Narumoto_Ito',
        '20230225_02_Matsushima_Ando'
    ]

    for video_name in video_name_lst:
        gtg = GrandTruthGenerator(VideoFrameLabelFactory.create_instance())
        gtg.dump_grand_truth_dataframe(video_name)
    return

    ds = VideoFrameLabelFactory.create_instance()
    pprint(ds)
    s = ds[video_name]
    pprint(ds[video_name][0].label_json)

    agg: FrameAggregationResult = s.aggregate_full()

    ordered_label = ['Stay', 'Play', 'Ready']

    result_lst = list(
        agg.extract_ordered_label_groups(
            label_order=[
                s.frame_label_name_list.index(ln)
                for ln in ordered_label
            ],
            predicate=lambda entry: entry.reliability > 0.5
        )
    )

    with storage.create_instance(
            domain='numpy_storage',
            entity=video_name,
            context='frames',
            mode='r',
    ) as snp_video_frame:
        assert isinstance(snp_video_frame, snp.NumpyStorage)
        src_fis = snp_video_frame.get_array('fi', fill_nan=-1)
        src_timestamps = snp_video_frame.get_array('timestamp')

        src_fi_ts_lut = {
            fi: ts
            for fi, ts in zip(src_fis, src_timestamps)
        }

        def find_nearest_timestamp(fi_target):
            abs_fi_delta = 0
            while True:
                for fi in [fi_target + abs_fi_delta, fi_target - abs_fi_delta]:
                    ts = src_fi_ts_lut.get(fi)
                    if ts is not None:
                        return ts
                abs_fi_delta += 1

    rows = []
    for i, result in enumerate(result_lst):
        fi_center_start = -1 if result.prev_frame is None else np.mean([
            result.prev_frame.fi_center,
            result.frames[0].fi_center
        ])
        fi_center_end = -1 if result.next_frame is None else np.mean([
            result.next_frame.fi_center,
            result.frames[-1].fi_center
        ])
        fi_centers = [f.fi_center for f in result.frames]
        labels = [f.label_index for f in result.frames]
        reliability = [f.reliability for f in result.frames]

        rows.append(
            dict(
                fi_center_start=fi_center_start,
                **{
                    label_name: fi
                    for label_name, fi in zip(
                        [ordered_label[i] for i in labels],
                        fi_centers
                    )
                },
                fi_center_end=fi_center_end,
                **{
                    'ts_' + label_name: find_nearest_timestamp(fi)
                    for label_name, fi in zip(
                        [ordered_label[i] for i in labels],
                        fi_centers
                    )
                },
                reliability=reliability
            )
        )

    df = pd.DataFrame(rows)
    print(df)


if __name__ == '__main__':
    main()
