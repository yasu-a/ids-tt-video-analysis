import os

__all__ = 'resolve_data_path',


def resolve_data_path(root_path, *args):
    return os.path.join('label_data', root_path, *args)
