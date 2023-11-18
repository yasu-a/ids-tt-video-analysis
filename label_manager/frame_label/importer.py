import json
import os.path
from typing import NamedTuple, Literal

import app_logging
from . import common
from . import util
from .json_struct import LabelJsonMeta, LabelJsonData, LabelJson
from .. import common as manager_common

logger = app_logging.create_logger(__name__)


class Importer(NamedTuple):
    source_json_path: str
    source_json_root: dict

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
    def meta(self) -> LabelJsonMeta:
        return LabelJsonMeta.from_json(
            {field_name: getattr(self, field_name) for field_name in LabelJsonMeta._fields}
        )

    @property
    def data(self) -> LabelJsonData:
        return LabelJsonData.from_json(self.source_json_root)

    @property
    def label_json(self) -> LabelJson:
        return LabelJson(
            meta=self.meta,
            labels=self.data
        )

    def label_json_root(self):
        return dict(
            meta=self.meta.to_json(),
            labels=self.source_json_root
        )

    @property
    def label_json_path(self):
        return manager_common.resolve_data_path(
            common.data_root_path,
            self.meta.video_name,
            self.meta.hash_digest + '.json'
        )

    def import_(self) -> Literal['already-exists', 'imported']:
        if os.path.exists(self.label_json_path):
            return 'already-exists'

        os.makedirs(os.path.dirname(self.label_json_path), exist_ok=True)

        with open(self.label_json_path, 'w') as f:
            json.dump(self.label_json_root(), f, indent=2, sort_keys=True)
        return 'imported'


def import_jsons(*source_path_lst):
    for source_path in source_path_lst:
        logger.info(f'Processing: {source_path!r}')
        importer = Importer.from_path(source_path)
        logger.info(f' - as {importer.label_json_path!r}')
        result = importer.import_()
        logger.info(f'Finished: {result}')