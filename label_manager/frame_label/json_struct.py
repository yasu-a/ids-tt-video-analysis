from typing import NamedTuple

import app_logging

logger = app_logging.create_logger(__name__)


class LabelJsonMeta(NamedTuple):
    video_name: str
    hash_digest: str

    @classmethod
    def from_json(cls, json_root: dict):
        if 'meta' in json_root:
            json_root = json_root['meta']

        return cls(**{field_name: json_root[field_name] for field_name in cls._fields})

    def to_json(self) -> dict:
        return self._asdict()


class __LabelJsonData(NamedTuple):
    markers: dict[int, str]
    tags: dict[int, list[str]]


class LabelJsonData(__LabelJsonData):
    def __new__(cls, markers, tags):
        markers = {int(k): markers[k] for k in sorted(markers.keys())}
        tags = {int(k): tags[k] for k in sorted(tags.keys())}

        if set(markers.keys()) != set(tags.keys()):
            logger.warning(f'Inconsistent frame indexes')

        # noinspection PyArgumentList
        return super().__new__(cls, markers, tags)

    @classmethod
    def from_json(cls, json_root: dict):
        if 'labels' in json_root:
            json_root = json_root['labels']

        return cls(**{field_name: json_root[field_name] for field_name in cls._fields})

    def to_json(self) -> dict:
        return self._asdict()


class LabelJson(NamedTuple):
    meta: LabelJsonMeta
    labels: LabelJsonData

    @classmethod
    def from_json(cls, json_root: dict):
        return cls(
            meta=LabelJsonMeta.from_json(json_root['meta']),
            labels=LabelJsonData.from_json(json_root['labels'])
        )

    def to_json(self) -> dict:
        return self._asdict()
