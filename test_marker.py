import collections
from pprint import pprint

from label_manager.frame_label.factory import VideoFrameLabelFactory


def main():
    ds = VideoFrameLabelFactory.create_instance()
    pprint(ds)
    s = ds['20230205_04_Narumoto_Harimoto']
    pprint(ds['20230205_04_Narumoto_Harimoto'][0].content)

    a = collections.defaultdict(list)
    for fn in s.frame_label_name_set:
        agg = s.aggregate(label_name=fn)
        print(agg.frame_index_center[agg.reliability > 0.5])
        print(agg.label_indexes[agg.reliability > 0.5])
        # print(fn, fis[(ars.n_sources >= 2) & (ars.reliability >= 0.5)])

    # pprint(cluster([groups[i]['Ready'] for i in range(len(groups))]))


if __name__ == '__main__':
    main()
