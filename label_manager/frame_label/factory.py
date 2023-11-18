import os

from . import common
from .sample_set import VideoFrameLabelSampleSet
from .. import common as manager_common


class VideoFrameLabelFactory:
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
        return VideoFrameLabelSampleSet(os.path.join(self.__root_path, video_name))

    def __repr__(self):
        return f'VFLFactory(root_path={self.__root_path!r})'
