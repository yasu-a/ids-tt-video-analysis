import copy
import os.path
from typing import Callable, Union, Any

import machine

DEFAULT_RAISE_ERROR = object()


class JsonPlaceholder:
    def __init__(self, param_name, *, default: Union[Callable, Any]):
        self.__param_name = param_name
        self.__default = default

    def create_replacement(self, params: dict[str, Any]):
        value = params.get(self.__param_name)
        if value is None:
            if self.__default is DEFAULT_RAISE_ERROR:
                raise ValueError('required parameter', self.__param_name)
            elif callable(self.__default):
                value = self.__default()
            else:
                value = self.__default
        else:
            params.pop(self.__param_name)
        return value


JSON_STRUCTURE = {
    1: {
        'markers': JsonPlaceholder('markers', default=lambda: {}),
        'tags': JsonPlaceholder('tags', default=lambda: {})
    },
    2: {
        'frames': JsonPlaceholder('frames', default=lambda: {}),
        'meta': {
            'json-version': 2,
            'author': {
                'system': machine.system,
                'node': machine.node,
                'hash-digest': machine.platform_hash_digest
            },
            'video-name': JsonPlaceholder('video_name', default=DEFAULT_RAISE_ERROR),
            'converted': JsonPlaceholder('converted', default=False)
        }
    }
}


def _replace_placeholder(json_structure, params):
    if isinstance(json_structure, list):
        return [_replace_placeholder(item, params) for item in json_structure]
    elif isinstance(json_structure, dict):
        return {k: _replace_placeholder(v, params) for k, v in json_structure.items()}
    elif isinstance(json_structure, JsonPlaceholder):
        placeholder = json_structure
        return placeholder.create_replacement(params)
    else:
        return json_structure


def create_default(version, params):
    json_structure = copy.deepcopy(JSON_STRUCTURE[version])

    json_root = _replace_placeholder(json_structure, params)
    if params:
        raise ValueError('unused parameters', params)

    return json_root


def _upgrade_1_to_2(src_json_path, src_json_root):
    # join markers and tags
    markers, tags = src_json_root['markers'], src_json_root['tags']
    frames = {
        str(k): dict(
            fi=int(k),  # frame index
            label=markers[k],
            tags=list(tags.get(k, []))
        )
        for k in markers.keys()
    }

    # extract video name from src_json_path
    _, file_name = os.path.split(src_json_path)
    file_name_except_suffix, _ = os.path.splitext(file_name)
    video_name = file_name_except_suffix

    # generate json version 2
    params = dict(
        frames=frames,
        video_name=video_name,
        converted=True
    )
    dst_json_root = create_default(2, params=params)

    return dst_json_root


def inspect_version(json_root):
    if 'meta' not in json_root:
        return 1
    else:
        return json_root['meta']['json-version']


def latest_json_version():
    return max(JSON_STRUCTURE.keys())


def convert(json_path, json_root, version_from=None, version_to=None):
    if version_from is None:
        version_from = inspect_version(json_root)
    if version_to is None:
        version_to = latest_json_version()

    if version_to < version_from:
        raise ValueError('tried to downgrade json format')
    elif version_to == version_from:
        return json_root
    elif version_to > version_from:
        # increment version by 1
        upgrade_handler_name = f'_upgrade_{version_from}_to_{version_from + 1}'
        upgrade_handler = globals()[upgrade_handler_name]
        assert callable(upgrade_handler), upgrade_handler
        upgraded_json_root = upgrade_handler(json_path, json_root)

        # retry converting
        return convert(
            json_path=json_path,
            json_root=upgraded_json_root,
            version_to=version_to
        )
