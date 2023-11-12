from contextlib import contextmanager
from typing import NamedTuple, Generic, TypeVar

from storage.route import create_data_path
from .impl import *
from .register import _find_storage_context, StorageContext

__all__ = 'NumpyStorage', 'create_instance'

NumpyStorageImplType = WriteModeNumpyStorageImpl  # TODO: add read-mode-impl

EntryNamedTupleGeneric = TypeVar('EntryNamedTupleGeneric', bound=NamedTuple)


class NumpyStorage(Generic[EntryNamedTupleGeneric]):
    def __init__(self, impl: NumpyStorageImplType, entry_named_tuple: type[EntryNamedTupleGeneric]):
        self.__impl = impl
        self.__entry_nt = entry_named_tuple

    def put_entry(self, i: int, entry: EntryNamedTupleGeneric) -> None:
        # noinspection PyProtectedMember
        self.__impl.put_entry(i, entry._asdict())

    def __setitem__(self, key, value):
        if not isinstance(key, int):
            raise TypeError()

        self.put_entry(key, value)

    def get_entry(self, i: int) -> dict[str, EntryNamedTupleGeneric]:
        return self.__entry_nt(**self.__impl.get_entry(i))

    def get_array(self, array_name):
        return self.__impl.get_array(array_name)

    def close(self):
        self.__impl.close()


@contextmanager
def create_instance(entity, context, *, mode, n_entries=None) -> NumpyStorage:
    if isinstance(context, str):
        storage_context = _find_storage_context(context)
    elif isinstance(context, StorageContext):
        storage_context = context
    else:
        raise TypeError('Parameter `context` must be a `str` or `StorageContext`')

    path = create_data_path(
        domain='numpy_storage',
        entity=entity,
        context=storage_context.name
    )

    if mode == 'w':
        impl = WriteModeNumpyStorageImpl(
            root_path=path,
            n_entries=n_entries,
            struct_definition=storage_context.struct_definition
        )
    elif mode == 'r':
        raise NotImplementedError()
    else:
        assert False, mode

    ns = NumpyStorage(
        impl=impl,
        entry_named_tuple=storage_context.entry_named_tuple
    )

    yield ns

    ns.close()
