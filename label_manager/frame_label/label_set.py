import os

from . import common
from .video_label_set import VideoFrameLabelSet
from .. import common as manager_common


class FrameLabelSet:
    def __init__(self, root_path):
        self.__root_path = root_path

    @classmethod
    def create_instance(cls):
        return cls(
            root_path=manager_common.resolve_data_path(common.data_root_path)
        )

    def video_names(self):
        return os.listdir(self.__root_path)

    def keys(self):
        return self.video_names()

    def __getitem__(self, video_name: str):
        return VideoFrameLabelSet(os.path.join(self.__root_path, video_name))

    def __repr__(self):
        return f'FrameLabelSet(root_path={self.__root_path!r})'
