import collections
import hashlib
import json
import os.path
import re
from pprint import pprint
from typing import Any, NamedTuple, Optional, Iterator

import numpy as np

from . import common

_root_path = 'markers'


def get_video_name_from_json_path(path):
    _, name = os.path.split(path)
    name, _ = os.path.splitext(name)
    m = re.match(r'[\S]+', name)
    if not m:
        raise ValueError('invalid json path', path)
    return m.group(0)


def load_md5_and_json(file_path) -> tuple[str, Any]:
    with open(file_path, 'r') as f:
        json_root = json.load(f)
        normalized_json_text = json.dumps(
            json_root,
            indent=2,
            sort_keys=True,
            ensure_ascii=True
        ).encode('latin-1')
        hash_value: str = hashlib.md5(normalized_json_text).hexdigest()
    return hash_value, json_root


class VideoMarkerEntry(NamedTuple):
    frame_index: int
    name: str
    tags: tuple[str]


class VideoMarker:
    def __init__(self, src_json_root):
        self.__json_root = self._normalize_json(src_json_root)
        self.__markers = self.__json_root['markers']
        self.__tags = self.__json_root['tags']
        self.__frame_indexes = np.array(sorted(self.__markers.keys()))
        self.__entries = {
            i: VideoMarkerEntry(
                frame_index=i,
                name=self.__markers[i],
                tags=self.__tags[i]
            ) for i in self.__frame_indexes
        }

    @classmethod
    def _normalize_json(cls, json_root):
        def parse_key_to_int(d: dict):
            try:
                return {int(k): v for k, v in d.items()}
            except ValueError:
                raise ValueError('invalid marker json format')

        markers = parse_key_to_int(json_root['markers'])
        tags = parse_key_to_int(json_root['tags'])

        # check frame_indexes
        if not set(markers.keys()) == set(tags.keys()):
            raise ValueError('invalid marker json format')

        # check json structure
        # TODO: check all structures
        if not all(isinstance(v, list) for v in tags.values()):
            raise ValueError('invalid marker json format')

        return dict(
            markers=markers,
            tags=tags
        )

    def dump(self, path):
        with open(path, 'w') as f:
            json.dump(self.__json_root, f, indent=2, sort_keys=True)

    @classmethod
    def load(cls, path):
        with open(path, 'r') as f:
            json_root = json.load(f)
        return cls(src_json_root=json_root)

    def __getitem__(self, frame_index) -> Optional[VideoMarkerEntry]:
        return self.__markers.get(frame_index)

    def keys(self) -> Iterator[int]:  # frame index
        yield from self.__entries.keys()

    def find_neighbour_frame_index(self, index):
        a = np.abs(self.__frame_indexes - index)
        return self.__frame_indexes[np.argmin(a)]


def import_json(json_path):
    video_name = get_video_name_from_json_path(json_path)

    dir_path = common.resolve_data_path(_root_path, video_name)
    os.makedirs(dir_path, exist_ok=True)

    hash_value, json_root = load_md5_and_json(json_path)
    file_path = os.path.join(dir_path, hash_value + '.json')

    if os.path.exists(file_path):
        return json_path, None

    VideoMarker(json_root).dump(file_path)

    return json_path, file_path


def iter_marker_dir_path_and_video_names():
    root_dir = common.resolve_data_path(_root_path)
    for video_name in os.listdir(root_dir):
        yield os.path.join(root_dir, video_name), video_name


class VideoMarkerSet(NamedTuple):
    markers: list[VideoMarker]

    @classmethod
    def create_full_set(cls) -> dict[str, 'VideoMarkerSet']:
        dct = collections.defaultdict(lambda: VideoMarkerSet(markers=[]))
        for dir_path, video_name in iter_marker_dir_path_and_video_names():
            for json_name in os.listdir(dir_path):
                json_path = os.path.join(dir_path, json_name)
                marker = VideoMarker.load(json_path)
                dct[video_name].markers.append(marker)
        return dct

    def calculate_meaningful_margin(self) -> float:
        margins = []
        for m in self.markers:
            frame_indexes = np.array(list(m.keys()))
            diff = np.diff(frame_indexes)
            margin = (diff.mean() - diff.std() * 2) / 2
            margins.append(margin)
        return min(margins)

    # FIXME: not working
    def classify(self):
        meaningful_margin = self.calculate_meaningful_margin()

        target_frame_indexes = sorted({
            frame_index
            for m in self.markers
            for frame_index in m.keys()
        })
        clst = collections.defaultdict(set)
        n_prev_targets = None
        i = 0
        while target_frame_indexes:
            target_fi = target_frame_indexes.pop()
            for m in self.markers:
                neighbour_fi = m.find_neighbour_frame_index(target_fi)
                if abs(neighbour_fi - target_fi) > meaningful_margin:
                    continue
                clst[target_fi].add(neighbour_fi)
                target_frame_indexes.remove(neighbour_fi)

            n_targets = len(target_frame_indexes)
            if n_prev_targets == n_targets:
                break
            n_prev_targets = n_targets

        return clst, target_frame_indexes


class MarkerDataSet:
    def __init__(self):
        self.__marker_set = VideoMarkerSet.create_full_set()
        for k, v in self.__marker_set.items():
            print(k)
            pprint(v.classify())
