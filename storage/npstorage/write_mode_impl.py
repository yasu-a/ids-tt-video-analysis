import json
import os
from abc import ABC

import numpy as np

import app_logging
from storage.context import context as _storage_context
from .impl_base import *

__all__ = 'WriteModeNumpyStorageImpl',


class WriteModeMetaStorageMixin(MetaStorageImplMixinBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.__meta_object = {}

    # def _meta_open(self):
    #     self._logger.debug('_meta_open: open \'%s\'', self._meta_path())
    #     with open(self._meta_path(), 'r') as f:
    #         self.__meta_object = json.load(f)

    def _meta_get_object(self) -> dict:
        return self.__meta_object

    def _meta_commit(self):
        self._logger.debug('_meta_commit: commit \'%s\'', self._meta_path())
        with open(self._meta_path(), 'w') as f:
            json.dump(self.__meta_object, f, indent=2, sort_keys=True)

    def _meta_close(self):
        self._logger.debug('_meta_close')
        self._meta_commit()


class WriteModeArrayStorageMixin(ArrayStorageMixinBase):
    def __init__(self, n_entries, **kwargs):
        if _storage_context.forbid_writing:
            raise ValueError('Writing is forbidden')

        super().__init__(**kwargs)

        self.__n_entries = n_entries

        self._logger.debug('_array__init__: initialized with n_entries=%d', n_entries)

        self.__arrays = {}  # array_name -> np.memmap
        self.__notified_entry_shape = {}  # array_name -> tuple

    def _array_count(self):
        return self.__n_entries

    def _array_open(self, array_name, shape):
        memmap_path = self._array_path(array_name)
        os.makedirs(os.path.dirname(memmap_path), exist_ok=True)
        a = np.memmap(
            filename=memmap_path,
            mode='w+',
            dtype=self._array_struct_dtype(array_name),
            shape=shape
        )
        self._logger.debug(
            '_array_open: %-20s created with\n'
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
            '_array_open: %-20s created np.memmap of size %d%cB',
            array_name,
            memmap_size_print,
            memmap_size_unit
        )

        init_value = self._array_struct_init_value(array_name)
        if init_value is not None:
            a[...] = init_value
            self._logger.info(
                '_array_open: %-20s initialized with init_value=%s',
                array_name,
                init_value
            )

        return a

    def _array_setup(self, array_name):
        self._logger.debug('_array_setup: %-20s', array_name)
        assert array_name in self.__notified_entry_shape, (array_name, self.__notified_entry_shape)
        entry_shape = self.__notified_entry_shape[array_name]
        shape = (self.__n_entries,) + entry_shape
        self.__arrays[array_name] = self._array_open(array_name, shape)

    def _array_notify_shape(self, array_name, entry_shape):
        self._logger.debug('_array_notify_shape: %-20s entry_shape=%s', array_name, entry_shape)
        self.__notified_entry_shape[array_name] = entry_shape

    def _array_ensure_initialized(self, array_name):
        if array_name not in self.__arrays:
            self._array_setup(array_name)

    def _array_memmap_data(self, array_name):
        self._array_ensure_initialized(array_name)
        return self.__arrays[array_name]

    def _array_memmap_status(self, array_name):
        self._array_ensure_initialized(self._status_array_name(array_name))
        return self.__arrays[self._status_array_name(array_name)]

    def _array_flush(self):
        for array_name, a in self.__arrays.items():
            self._logger.debug('_array_flush: %-20s', array_name)
            a.flush()

    def _array_close(self):
        for array_name, a in self.__arrays.items():
            self._logger.info('_array_close: %-20s', array_name)
            # noinspection PyProtectedMember
            a._mmap.close()


class WriteModeNumpyStorageImpl(
    ModifyCounterMixin,
    WriteModeArrayStorageMixin,
    WriteModeMetaStorageMixin,
    NumpyStorageImplBase,
    ABC
):
    _COMMIT_INTERVAL = 256

    def __init__(self, **kwargs):
        logger = app_logging.create_logger(__name__)
        super().__init__(logger=logger, **kwargs)

    def commit(self):
        if self._modcnt_n_left_modification() == 0:
            return

        self._array_flush()
        self._modcnt_committed()

    def _commit_if_possible(self):
        if self._modcnt_n_left_modification() > self._COMMIT_INTERVAL:
            self.commit()

    def _update_or_validate_entry_shape(self, array_name, entry_obj):
        # retrieve entry shape
        if hasattr(entry_obj, 'shape'):
            entry_shape = entry_obj.shape
        else:
            entry_shape = ()

        # get meta
        meta = self._meta_get_object()

        shape_meta = meta.get('shape')
        if shape_meta is None:
            shape_meta = meta['shape'] = {}

        specific_shape = shape_meta.get(array_name)
        if specific_shape is None:
            shape_meta[array_name] = (self._array_count(),) + entry_shape
            shape_meta[self._status_array_name(array_name)] = (self._array_count(),)
            self._meta_commit()
            self._array_notify_shape(array_name, entry_shape)
            self._array_notify_shape(self._status_array_name(array_name), ())
        else:
            if specific_shape[1:] != entry_shape:
                raise ValueError('Invalid shape', array_name, entry_shape)

    def put_entry(self, i: int, mapping: dict[str, EntryInputType]) -> None:
        for array_name, obj in mapping.items():
            if array_name not in self._array_struct_array_names():
                raise ValueError('Invalid array name', array_name)
            self._update_or_validate_entry_shape(array_name, obj)
            self._array_memmap_data(array_name)[i, ...] = obj
            self._array_memmap_status(array_name)[i] = self.STATUS_VALID
        self._modcnt_count()
        self._commit_if_possible()

    def get_entry(self, i: int) -> dict[str, EntryOutputType]:
        raise ValueError('Operation not supported')

    def get_array(self, array_name, fill_nan=np.nan) -> EntryOutputType:
        raise ValueError('Operation not supported')

    def get_status(self, array_name) -> EntryOutputType:
        raise ValueError('Operation not supported')

    def get_array_names(self) -> frozenset[str]:
        return self._array_struct_array_names()

    def count(self) -> int:
        return self._array_count()

    def close(self) -> None:
        self._array_close()
        self._meta_close()
