import json
import os
from abc import ABC

import numpy as np

import app_logging
from .impl_base import *

__all__ = 'ReadModeNumpyStorageImpl',


class ReadModeMetaStorageMixin(MetaStorageImplMixinBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.__meta_object = {}

    def _meta_open(self):
        self._logger.debug('_meta_open: open \'%s\'', self._meta_path())
        with open(self._meta_path(), 'r') as f:
            self.__meta_object = json.load(f)

    def _meta_get_object(self) -> dict:
        return self.__meta_object

    def _meta_close(self):
        self._logger.debug('_meta_close')


class ReadModeArrayStorageMixin(ArrayStorageMixinBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._logger.debug('_array__init__: initialized')

        self.__arrays = {}  # array_name -> np.memmap
        self.__notified_array_shape = {}  # array_name -> tuple

    def _array_open(self, array_name, shape):
        memmap_path = self._array_path(array_name)
        os.makedirs(os.path.dirname(memmap_path), exist_ok=True)
        a = np.memmap(
            filename=memmap_path,
            mode='r',
            dtype=self._array_struct_dtype(array_name),
        )
        a = a.reshape(shape)
        self._logger.debug(
            '_array_open: %-20s loaded with\n'
            ' - filename=\'%s\'\n'
            ' - mode=\'%s\'\n'
            ' - dtype=%s\n'
            ' - shape=%s',
            array_name,
            a.filename,
            a.mode,
            a.dtype,
            a.shape
        )

        memmap_size = os.stat(memmap_path).st_size
        if memmap_size > 1e+7:
            memmap_size_print = memmap_size / 1000 / 1000
            memmap_size_unit = 'M'
        else:
            memmap_size_print = memmap_size / 1000
            memmap_size_unit = 'K'
        self._logger.info(
            '_array_open: %-20s loaded np.memmap of size %d%cB',
            array_name,
            memmap_size_print,
            memmap_size_unit
        )

        return a

    def _array_setup(self, array_name):
        self._logger.debug('_array_setup: %-20s', array_name)
        assert array_name in self.__notified_array_shape, (array_name, self.__notified_array_shape)
        shape = self.__notified_array_shape[array_name]
        self.__arrays[array_name] = self._array_open(array_name, shape)

    def _array_notify_shape(self, array_name, array_shape):
        self._logger.debug('_array_notify_shape: %-20s shape=%s', array_name, array_shape)
        self.__notified_array_shape[array_name] = array_shape

    def _array_ensure_initialized(self, array_name):
        if array_name not in self.__arrays:
            self._array_setup(array_name)

    def _array_memmap_data(self, array_name):
        self._array_ensure_initialized(array_name)
        return self.__arrays[array_name]

    def _array_memmap_status(self, array_name):
        self._array_ensure_initialized(self._status_array_name(array_name))
        return self.__arrays[self._status_array_name(array_name)]

    def _array_close(self):
        for array_name, a in self.__arrays.items():
            self._logger.info('_array_close: %-20s', array_name)
            # noinspection PyProtectedMember
            a._mmap.close()


class ReadModeNumpyStorageImpl(
    ReadModeArrayStorageMixin,
    ReadModeMetaStorageMixin,
    NumpyStorageImplBase,
    ABC
):
    def __init__(self, **kwargs):
        logger = app_logging.create_logger(__name__)
        super().__init__(logger=logger, **kwargs)
        self._meta_open()
        for array_name in self._array_struct_array_names():
            self.__notify_array_shape(array_name)

    def put_entry(self, i: int, mapping: dict[str, EntryInputType]) -> None:
        raise ValueError('Operation not supported')

    def __notify_array_shape(self, array_name):
        self._array_notify_shape(
            array_name,
            self._meta_get_object()['shape'][array_name]
        )
        self._array_notify_shape(
            self._status_array_name(array_name),
            self._meta_get_object()['shape'][self._status_array_name(array_name)]
        )

    def get_entry(self, i: int) -> dict[str, EntryOutputType]:
        entry = {}
        for array_name in self._array_struct_array_names():
            self.__notify_array_shape(array_name)
            obj = self._array_memmap_data(array_name)[i, ...]
            status = self._array_memmap_status(array_name)[i]
            if status == self.STATUS_INVALID:
                obj = None
            entry[array_name] = obj
        return entry

    def __get_array_raw(self, array_name):
        self.__notify_array_shape(array_name)
        obj = self._array_memmap_data(array_name)[:, ...]
        return obj

    def get_array(self, array_name, fill_nan=np.nan) -> EntryOutputType:
        obj = self.__get_array_raw(array_name)
        status = self._array_memmap_status(array_name)
        obj[status == self.STATUS_INVALID] = fill_nan
        return obj

    def get_status(self, array_name) -> EntryOutputType:
        return self._array_memmap_status(array_name)

    def get_array_names(self) -> frozenset[str]:
        return self._array_struct_array_names()

    def count(self) -> int:
        array_name_sampled = next(iter(self._array_struct_array_names()))
        return len(self.__get_array_raw(array_name_sampled))

    def close(self) -> None:
        self._array_close()
        self._meta_close()
