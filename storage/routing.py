import os
from typing import NamedTuple

from config import config


def root_location():
    return os.path.normpath(config.data_location)


class StoragePath(NamedTuple):
    domain: str
    entity: str
    context: str
    args: tuple[str, ...] = ()

    @property
    def path(self):
        return os.path.join(
            root_location(),
            self.domain,
            self.entity,
            self.context,
            *self.args
        )

    def list_sizes(self):
        for root, dir_names, file_names in os.walk(self.path):
            for file_name in file_names:
                file_path = os.path.join(root, file_name)
                yield file_path, os.stat(file_path).st_size

    @property
    def total_size(self):
        return sum(size for path, size in self.list_sizes())

    @classmethod
    def list_storages(cls):
        def it():
            for root, dir_names, file_names in os.walk(root_location()):
                for file_name in file_names:
                    if file_name != 'meta.json':
                        continue
                    # first element is always empty
                    _, domain, entity, context, *args \
                        = os.path.normpath(root)[len(root_location()):].split(os.sep)
                    yield StoragePath(
                        domain=domain,
                        entity=entity,
                        context=context,
                        args=args
                    )
                    dir_names.clear()

        return list(it())
