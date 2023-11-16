from pprint import pprint

import numpy as np

import labels.marker

ds = labels.marker.VideoMarkerSet.create_full_set()
pprint(ds)
_, marker_set = ds.popitem()


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


def cluster(groups: list[np.ndarray]):
    n_groups = len(groups)

    for gi in range(n_groups):
        groups[gi] = np.sort(groups[gi])

    pprint(groups)

    margin = min(calculate_margin(np.diff(a), p_thresh=0.9999) for a in groups)

    cluster_labels = []
    for gi in range(n_groups):
        cluster_labels.append(np.array([-1] * len(groups[gi])))

    new_cluster_label = 0

    pointers = [0] * n_groups
    while any(p < len(groups[i]) for i, p in enumerate(pointers)):
        row_data = np.array([
            groups[i][p] if p < len(groups[i]) else None
            for i, p in enumerate(pointers)
        ])

        # grouping row
        adj_matrix = [
            [
                i != j and
                row_data[i] is not None and
                row_data[j] is not None and
                abs(row_data[i] - row_data[j]) < margin
                for i in range(n_groups)
            ]
            for j in range(n_groups)
        ]

        item_labels = [-1] * n_groups
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
                    for j in range(n_groups)
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
            cluster_labels[gi][pointers[gi]] = new_cluster_label
            pointers[gi] += 1
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
