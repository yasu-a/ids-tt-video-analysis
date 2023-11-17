import functools
import os

from .video_label_entry import VideoFrameLabel


class VideoFrameLabelSetMixin:
    def __len__(self): ...

    def __getitem__(self, i: int) -> VideoFrameLabel: ...

    # TODO: implement me


class VideoFrameLabelSet(VideoFrameLabelSetMixin):
    def __init__(self, path: str):
        self.__path = path

    @functools.cache
    def list_json_names(self):
        return tuple(sorted(os.listdir(self.__path)))

    def __len__(self):
        return len(self.list_json_names())

    def __getitem__(self, i: int) -> VideoFrameLabel:
        return VideoFrameLabel(os.path.join(self.__path, self.list_json_names()[i]))

    def __repr__(self):
        return f'FrameLabelSet(path={self.__path!r})'
