from pprint import pprint

import numpy as np

import label_manager.frame_label

ds = label_manager.marker.VideoMarkerSet.create_full_set()
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


def cluster(arrays_: list[np.ndarray]) -> list[np.ndarray]:
    margin = min(calculate_margin(np.diff(a), p_thresh=0.9999) for a in arrays_)

    points = np.concatenate(arrays_)[:, None]

    from sklearn.cluster import DBSCAN

    labels = DBSCAN(eps=margin, min_samples=2).fit_predict(points)

    label_lst = []
    for a in arrays_:
        n = len(a)
        label_lst.append(labels[:n])
        labels = labels[n:]

    return [np.array(a) for a in label_lst]


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
