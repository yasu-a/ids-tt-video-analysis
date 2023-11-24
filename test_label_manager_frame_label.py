from pprint import pprint

from label_manager.frame_label.factory import VideoFrameLabelFactory
from label_manager.frame_label.sample_set import FrameAggregationResult


def main():
    ds = VideoFrameLabelFactory.create_instance()
    pprint(ds)
    s = ds['20230219_03_Narumoto_Ito']
    pprint(ds['20230219_03_Narumoto_Ito'][0].content)

    agg: FrameAggregationResult = s.aggregate_full()

    pprint(
        list(
            agg.extract_ordered_label_groups(
                label_order=[
                    s.frame_label_name_list.index(ln)
                    for ln in ['Stay', 'Play', 'Ready']
                ],
                predicate=lambda entry: entry.reliability > 0.5
            )
        )
    )

    # print(agg.frame_index_center[agg.reliability > 0.5])
    # print(agg.label_indexes[agg.reliability > 0.5])
    # print(fn, fis[(ars.n_sources >= 2) & (ars.reliability >= 0.5)])

    # pprint(cluster([groups[i]['Ready'] for i in range(len(groups))]))


if __name__ == '__main__':
    main()
