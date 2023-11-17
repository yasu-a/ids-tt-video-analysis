import json

from .json_struct import LabelJson


class VideoFrameLabelMixin:
    content: LabelJson = ...

    # TODO: implement me


class VideoFrameLabel(VideoFrameLabelMixin):
    def __init__(self, json_path):
        self.__json_path = json_path

        with open(json_path, 'r') as f:
            self.__content = LabelJson.from_json(json.load(f))

    @property
    def content(self) -> LabelJson:
        return self.__content

    def __repr__(self):
        return f'FrameLabelSet(json_path={self.__json_path!r})'
