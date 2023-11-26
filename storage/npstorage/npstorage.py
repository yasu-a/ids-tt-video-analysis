from typing import NamedTuple, Generic, TypeVar, Union

import numpy as np

from storage.routing import StoragePath
from .impl import *
from .impl_base import NumpyStorageImplBase
from .register import _find_storage_context, StorageContext

__all__ = 'NumpyStorage', '_create_domain_instance', 'STATUS_INVALID', 'STATUS_VALID'

NumpyStorageImplType = Union[WriteModeNumpyStorageImpl, ReadModeNumpyStorageImpl]

EntryNamedTupleGeneric = TypeVar('EntryNamedTupleGeneric', bound=NamedTuple)

STATUS_INVALID = NumpyStorageImplBase.STATUS_INVALID
STATUS_VALID = NumpyStorageImplBase.STATUS_VALID


class NumpyStorage(Generic[EntryNamedTupleGeneric]):
    def __init__(
            self,
            impl: NumpyStorageImplType,
            entry_named_tuple: type[EntryNamedTupleGeneric]
    ):
        self.__impl = impl
        self.__entry_nt = entry_named_tuple

    def put_entry(self, i: int, entry: EntryNamedTupleGeneric) -> None:
        # noinspection PyProtectedMember
        self.__impl.put_entry(i, entry._asdict())

    def __setitem__(self, key, value):
        if not isinstance(key, int):
            raise TypeError()

        self.put_entry(key, value)

    def get_entry(self, i: int) -> EntryNamedTupleGeneric:
        return self.__entry_nt(**self.__impl.get_entry(i))

    def get_array(self, array_name, fill_nan=np.nan) -> EntryOutputType:
        return self.__impl.get_array(array_name, fill_nan=fill_nan)

    def get_status(self, array_name) -> EntryOutputType:
        return self.__impl.get_status(array_name)

    def get_array_names(self) -> frozenset[str]:
        return self.__impl.get_array_names()

    def count(self) -> int:
        return self.__impl.count()

    def close(self):
        self.__impl.close()


def _create_domain_instance(entity, context, *, mode, n_entries=None):
    if isinstance(context, str):
        storage_context = _find_storage_context(context)
    elif isinstance(context, StorageContext):
        storage_context = context
    else:
        raise TypeError('Parameter `context` must be a `str` or `StorageContext`')

    path = StoragePath(
        domain='numpy_storage',
        entity=entity,
        context=storage_context.name
    ).path

    if mode == 'w':
        impl = WriteModeNumpyStorageImpl(
            root_path=path,
            n_entries=n_entries,
            struct_definition=storage_context.struct_definition
        )
    elif mode == 'r':
        impl = ReadModeNumpyStorageImpl(
            root_path=path,
            struct_definition=storage_context.struct_definition
        )
    else:
        assert False, mode

    return NumpyStorage(
        impl=impl,
        entry_named_tuple=storage_context.entry_named_tuple
    )
