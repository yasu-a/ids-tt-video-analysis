import json
import os.path
from typing import NamedTuple, Literal

import app_logging
from . import common
from . import util
from .. import common as labels_common

logger = app_logging.create_logger(__name__)


class JsonMeta(NamedTuple):
    video_name: str
    hash_digest: str

    @classmethod
    def from_json(cls, json_root: dict):
        if 'meta' in json_root:
            json_root = json_root['meta']

        return cls(**{field_name: json_root[field_name] for field_name in cls._fields})

    def to_json(self) -> dict:
        return self._asdict()


class Importer(NamedTuple):
    source_json_path: str
    source_json_root: str

    @classmethod
    def from_path(cls, source_json_path):
        with open(source_json_path, 'r') as f:
            source_json_root = json.load(f)
        return cls(
            source_json_path=source_json_path,
            source_json_root=source_json_root
        )

    @property
    def video_name(self):
        return util.extract_video_name_from_json_path(self.source_json_path)

    @property
    def hash_digest(self):
        return util.json_to_md5_digest(self.source_json_root)

    @property
    def meta(self) -> JsonMeta:
        return JsonMeta.from_json(
            {field_name: getattr(self, field_name) for field_name in JsonMeta._fields}
        )

    def label_data_json_root(self):
        return dict(
            meta=self.meta.to_json(),
            labels=self.source_json_root
        )

    @property
    def label_data_json_path(self):
        return labels_common.resolve_data_path(
            common.data_root_path,
            self.meta.video_name,
            self.meta.hash_digest + '.json'
        )

    def import_(self) -> Literal['already-exists', 'imported']:
        if os.path.exists(self.label_data_json_path):
            return 'already-exists'

        os.makedirs(os.path.dirname(self.label_data_json_path), exist_ok=True)

        with open(self.label_data_json_path, 'w') as f:
            json.dump(self.label_data_json_root(), f, indent=2, sort_keys=True)
        return 'imported'


def import_jsons(*source_path_lst):
    for source_path in source_path_lst:
        logger.info(f'Processing: {source_path!r}')
        importer = Importer.from_path(source_path)
        logger.info(f' - as {importer.label_data_json_path!r}')
        result = importer.import_()
        logger.info(f'Finished: {result}')
