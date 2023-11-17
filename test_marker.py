from pprint import pprint

import numpy as np

from label_manager.frame_label.label_set import FrameLabelSet


def main():
    ds = FrameLabelSet.create_instance()
    pprint(ds)
    s = ds['20230205_04_Narumoto_Harimoto']
    pprint(ds['20230205_04_Narumoto_Harimoto'][0].content)

    ordered_keys = sorted(marker_set.marker_set)
    pprint(ordered_keys)
    # video_marker_index:int -> marker_name:str -> frame_index_array: int
    groups: list[dict[str, np.ndarray]] = [
        video_marker.frame_indexes_grouped_by_marker_name()
        for video_marker in ordered_keys
    ]

    # pprint(cluster([groups[i]['Ready'] for i in range(len(groups))]))


if __name__ == '__main__':
    main()
