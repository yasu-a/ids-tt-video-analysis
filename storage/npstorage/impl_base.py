import functools
import os
from typing import Any, Union

import numpy as np

import app_logging

__all__ = (
    'NumpyStoragePrototype',
    'NumpyStorageImplBase',
    'MetaStorageImplMixinBase',
    'ArrayStorageMixinBase',
    'ModifyCounterMixin',
    'EntryInputType',
    'EntryOutputType'
)


class NumpyStoragePrototype:
    _root_path: str
    _struct: dict[str, dict[str, Any]]
    _logger: app_logging.Logger

    @classmethod
    def _status_array_name(cls, array_name) -> str: ...


EntryType = Union[int, float, np.ndarray]
EntryOutputType = EntryType
EntryInputType = Union[EntryType, list, tuple]


class NumpyStorageImplBase(NumpyStoragePrototype):
    STATUS_INVALID = 0
    STATUS_VALID = 1

    @classmethod
    def _status_array_name(cls, array_name):
        return f'__status_{array_name}__'

    @classmethod
    def _parse_struct_definition(cls, struct_definition) -> dict[str, dict[str, Any]]:
        structure = {}
        for array_name, definition in struct_definition.items():
            assert all(k in definition.keys() for k in {'dtype'}), \
                (struct_definition, array_name)
            assert all(k in {'dtype', 'init_value'} for k in definition.keys()), \
                (struct_definition, array_name)
            structure[array_name] = definition
            structure[cls._status_array_name(array_name)] = dict(
                dtype=np.uint8,
                init_value=cls.STATUS_INVALID
            )
        return structure

    def __init__(self, root_path, struct_definition, logger, **kwargs):
        assert kwargs == {}, (kwargs, type(self).mro())

        self._root_path = root_path
        self.__struct_definition = struct_definition
        self._logger = logger

        os.makedirs(self._root_path, exist_ok=True)

    @functools.cached_property
    def _struct(self):
        return self._parse_struct_definition(self.__struct_definition)

    def put_entry(self, i: int, mapping: dict[str, EntryInputType]) -> None:
        raise NotImplementedError()

    def get_entry(self, i: int) -> dict[str, EntryOutputType]:
        raise NotImplementedError()

    def get_array(self, array_name, fill_nan=np.nan) -> EntryOutputType:
        raise NotImplementedError()

    def count(self) -> int:
        raise NotImplementedError()

    def close(self) -> None:
        raise NotImplementedError()


class MetaStorageImplMixinBase(NumpyStoragePrototype):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _meta_path(self):
        return os.path.join(self._root_path, 'meta.json')


class ArrayStorageMixinBase(NumpyStoragePrototype):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _array_path(self, array_name):
        return os.path.join(self._root_path, 'arrays', array_name)

    def _array_struct_dtype(self, array_name):
        return self._struct[array_name]['dtype']

    def _array_struct_init_value(self, array_name):
        return self._struct[array_name].get('init_value')

    def _array_struct_array_names(self):
        return self._struct.keys()


# TODO: make mixin a class and apply it for meta-storages and array-storages
class ModifyCounterMixin:
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.__modification_count = 0
        self.__modification_count_on_last_commit = 0

    def _modcnt_count(self):
        self.__modification_count += 1

    def _modcnt_committed(self):
        self.__modification_count_on_last_commit = self.__modification_count

    def _modcnt_n_left_modification(self):
        return self.__modification_count - self.__modification_count_on_last_commit
