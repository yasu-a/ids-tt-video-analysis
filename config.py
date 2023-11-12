import functools
import json
import platform

__all__ = 'config',

_device_name = platform.node()


class Config:
    def __init__(self, json_path):
        with open(json_path, 'r') as f:
            self.__data = json.load(f)

    @property
    def _data(self):
        return self.__data

    @property
    def data_location(self):
        return self['data-location']

    @property
    def video_location(self):
        return self['video-location']

    @property
    def default_video_name(self):
        return self['default-video-name']

    @functools.cache
    def _values_current_context(self, name=None):
        name = name or _device_name

        current_data = self.__data['env-config'].get(name)
        if current_data is None:
            raise ValueError(f'context for {name!r} not found; current device is {_device_name!r}')

        inherit_name = current_data.get('.inherit')
        if inherit_name is None:
            base_data = {}
        else:
            base_data = self._values_current_context(name=inherit_name)
            current_data.pop('.inherit')

        return base_data | current_data

    def __getitem__(self, item):
        return self._values_current_context()[item]


config = Config(json_path=r'./dataset_config.json')

if __name__ == '__main__':
    from pprint import pprint

    print(_device_name)
    # noinspection PyProtectedMember
    pprint(config._values_current_context('DESKTOP-C4V3D9S'))
