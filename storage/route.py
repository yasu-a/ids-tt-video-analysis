import os

from config import config


def create_data_path(domain, entity, context, *args) -> str:
    path = os.path.join(config.data_location, domain, entity, context, *args)
    return path
