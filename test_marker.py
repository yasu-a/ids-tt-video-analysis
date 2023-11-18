from pprint import pprint

from label_manager.frame_label.factory import VideoFrameLabelFactory


def main():
    ds = VideoFrameLabelFactory.create_instance()
    pprint(ds)
    s = ds['20230205_04_Narumoto_Harimoto']
    pprint(ds['20230205_04_Narumoto_Harimoto'][0].content)

    for fn in s.frame_label_name_set:
        fis = s.agg_frame_indexes(label_name=fn)
        ars = s.agg_results(label_name=fn)
        print(fn, fis[(ars.n_sources >= 2) & (ars.reliability >= 0.5)])

    # pprint(cluster([groups[i]['Ready'] for i in range(len(groups))]))


if __name__ == '__main__':
    main()
