from contextlib import contextmanager

__all__ = 'create_instance',


@contextmanager
def create_instance(domain, entity, context, *, mode, n_entries=None):
    assert domain == 'numpy_storage'
    from . import npstorage
    # noinspection PyProtectedMember
    storage = npstorage._create_domain_instance(
        entity,
        context,
        mode=mode,
        n_entries=n_entries
    )
    try:
        yield storage
    finally:
        storage.close()
