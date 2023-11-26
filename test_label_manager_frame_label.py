from pprint import pprint

import numpy as np
import pandas as pd

import storage
import storage.npstorage as snp
from label_manager.frame_label.factory import VideoFrameLabelFactory
from label_manager.frame_label.sample_set import FrameAggregationResult

storage.context.forbid_writing = True

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.expand_frame_repr', False)


def main():
    video_name = '20230219_03_Narumoto_Ito'
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
