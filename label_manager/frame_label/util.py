import hashlib
import json
import os.path
import re
from typing import Any

import numpy as np


def extract_video_name_from_json_path(path):
    _, name = os.path.split(path)
    name, _ = os.path.splitext(name)
    m = re.match(r'\S+', name)
    if not m:
        raise ValueError('invalid json path', path)
    return m.group(0)


def normalized_json_text(json_root):
    norm = json.dumps(
        json_root,
        indent=2,
        sort_keys=True,
        ensure_ascii=True
    ).encode('latin-1')
    return norm


def json_to_md5_digest(json_root):
    norm = normalized_json_text(json_root)
    hash_value: str = hashlib.md5(norm).hexdigest()
    return hash_value


def calculate_margin(diff: np.ndarray, p_thresh=99.99 / 100) -> float:
    """
    ほどんどの点がある距離以上の間隔を隔てて並ぶ１次元空間内の点の集合について，隣接する点どうしの
    距離のサンプル`diff`を統計的に処理し，隣接する点どうしを`p_thresh`の確率で分離するための，
    隣接する点どうしの距離が満たすべき最大の間隔を計算する。
    :param diff: 隣接する点どうしの距離
    :param p_thresh: 分離確率
    :return: `p_thresh`の確率で隣接する点どうしを分離できる最小距離
    """
    assert np.all(diff >= 0), diff

    hist = np.zeros(diff.max(initial=0) + 1)
    x = np.arange(diff.max(initial=0) + 1)
    indexes, counts = np.unique(diff, return_counts=True)
    hist[indexes] = counts

    from scipy.stats import gaussian_kde
    kernel = gaussian_kde(diff)

    from scipy.special import ndtr
    cdf = np.array(
        list(ndtr(np.ravel(item - kernel.dataset) / kernel.factor).mean() for item in x)
    )
    thresh_index = x[cdf < 1 - p_thresh].max(initial=0)
    margin = thresh_index

    return margin


def cluster(arrays_: list[np.ndarray]) -> tuple[list[np.ndarray], dict[str, Any]]:
    margin = min(calculate_margin(np.diff(np.sort(a)), p_thresh=0.9999) for a in arrays_)

    points = np.concatenate(arrays_)[:, None]

    from sklearn.cluster import DBSCAN
    labels = DBSCAN(eps=margin, min_samples=2).fit_predict(points)

    label_lst = []
    for a in arrays_:
        n = len(a)
        label_lst.append(labels[:n])
        labels = labels[n:]

    return [np.array(a) for a in label_lst], dict(margin=margin)
