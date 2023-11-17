import functools
from pprint import pprint

import numpy as np

import labels.marker

ds = labels.marker.VideoMarkerSet.create_full_set()
pprint(ds)
marker_set = ds['20230205_04_Narumoto_Harimoto']


def calculate_margin(diff: np.ndarray, p_thresh=99.99 / 100) -> float:
    """
    ほどんどの点がある距離以上の間隔を隔てて並ぶ１次元空間内の点の集合について，隣接する点どうしの
    距離のサンプル`diff`を統計的に処理し，隣接する点どうしを`p_thresh`の確率で分離するための，
    隣接する点どうしの距離が満たすべき最大の間隔を計算する。
    :param diff: 隣接する点どうしの距離
    :param p_thresh: 分離確率
    :return: `p_thresh`の確率で隣接する点どうしを分離できる最小距離
    """
    hist = np.zeros(diff.max() + 1)
    x = np.arange(diff.max() + 1)
    indexes, counts = np.unique(diff, return_counts=True)
    hist[indexes] = counts

    from scipy.stats import gaussian_kde
    kernel = gaussian_kde(diff)

    from scipy.special import ndtr
    cdf = np.array(
        list(ndtr(np.ravel(item - kernel.dataset) / kernel.factor).mean() for item in x)
    )
    thresh_index = x[cdf < 1 - p_thresh].max(0)
    margin = thresh_index

    return margin


class ParallelArrayReader:
    def __init__(self, arrays: list[np.ndarray]):
        self.__arrays = [np.sort(a) for a in arrays]
        self.__pointers = np.zeros(len(arrays), dtype=int)

    @property
    def n_arrays(self):
        return len(self.__arrays)

    def iter_array_index(self):
        return range(self.n_arrays)

    @functools.cached_property
    def array_length(self):
        return np.array([len(a) for a in self.__arrays])

    def valid_array_mask(self):
        return self.__pointers < self.array_length

    def parallel_data(self):
        valid_array_mask = self.valid_array_mask()
        return np.array([
            self.__arrays[g][i] if valid_array_mask[g] else np.nan
            for g, i in enumerate(self.__pointers)
        ])

    @property
    def pointers(self):
        return self.__pointers

    def increment(self, indexes):
        self.__pointers[indexes] += 1


def cluster(arrays_: list[np.ndarray]):
    reader = ParallelArrayReader(arrays_)

    pprint(arrays_)

    margin = min(calculate_margin(np.diff(a), p_thresh=0.9999) for a in arrays_)

    cluster_labels = []
    for gi in reader.iter_array_index():
        cluster_labels.append(np.array([-1] * len(arrays_[gi])))

    new_cluster_label = 0

    while np.all(reader.valid_array_mask()):
        row_data = reader.parallel_data()

        # grouping row
        adj_matrix = [
            [
                i != j and
                row_data[i] is not None and
                row_data[j] is not None and
                abs(row_data[i] - row_data[j]) < margin
                for i in reader.iter_array_index()
            ]
            for j in reader.iter_array_index()
        ]

        item_labels = [-1] * reader.n_arrays
        while True:
            # pop non-labeled group index
            non_labeled_group_index = {i for i, lbl in enumerate(item_labels) if lbl < 0}

            # if all items are labeled
            if not non_labeled_group_index:
                break

            start = non_labeled_group_index.pop()

            # find connected
            stack = [start]
            history = []
            while stack:
                gi = stack.pop()
                history.append(gi)
                connected = [
                    j
                    for j in reader.iter_array_index()
                    if adj_matrix[gi][j] and j not in history
                ]
                stack += connected

            # new_label found indexes
            new_label = max(item_labels) + 1
            for gi in history:
                item_labels[gi] = new_label

        # take group with minimum centroid
        # generate group iterator
        group_it = [
            [gi for gi, lbl in enumerate(item_labels) if lbl == target_lbl]
            for target_lbl in sorted(set(item_labels))
        ]

        # find group indexes with minimum centroid
        best_gis = None
        best_centroid = None
        for gis in group_it:
            if gis[0] is None:
                assert len(gis) == 1, gis
                continue
            centroid = row_data[gis].mean()
            if best_centroid is None or best_centroid > centroid:
                best_gis = gis
                best_centroid = centroid

        assert best_gis is not None, item_labels

        # set cluster labels on items in best_gis
        for gi in best_gis:
            cluster_labels[gi][reader.pointers[gi]] = new_cluster_label
        reader.increment(best_gis)
        new_cluster_label += 1

    return cluster_labels


def main():
    ordered_keys = sorted(marker_set.marker_set)
    pprint(ordered_keys)
    # video_marker_index:int -> marker_name:str -> frame_index_array: int
    groups: list[dict[str, np.ndarray]] = [
        video_marker.frame_indexes_grouped_by_marker_name()
        for video_marker in ordered_keys
    ]

    pprint(cluster([groups[i]['Ready'] for i in range(len(groups))]))


if __name__ == '__main__':
    main()
