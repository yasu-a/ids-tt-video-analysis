from typing import Any, NamedTuple

__all__ = (
    'StorageContext',
    'register_storage_context',
    '_find_storage_context',
    'list_storage_context'
)


class StorageContext(NamedTuple):
    name: str
    struct_definition: dict[str, dict[str, Any]]
    entry_named_tuple: type[NamedTuple]


_registrations: dict[str, StorageContext] = {}  # name -> StorageContext


def register_storage_context(storage_context: StorageContext):
    global _registrations
    assert storage_context.name not in _registrations, set(_registrations.keys())
    _registrations[storage_context.name] = storage_context


def _find_storage_context(name: str):
    return _registrations[name]


def list_storage_context():
    return list(_registrations.keys())
