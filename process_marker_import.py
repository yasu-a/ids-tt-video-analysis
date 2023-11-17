import argparse
import glob
import os.path

import process
from label_manager.frame_label.importer import import_jsons


class ProcessStageMarkerImport(process.ProcessStage):
    NAME = 'marker-import'
    ALIASES = 'mi',

    @classmethod
    def customize_parser(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument('json_paths', type=str, nargs='+')

    def __init__(self, json_paths: list[str]):
        paths = []
        for maybe_path in json_paths:
            if os.path.exists(maybe_path):
                paths.append(maybe_path)
            else:
                detected_paths = glob.glob(maybe_path, recursive=True)
                if not detected_paths:
                    print(f'Warning: nothing extracted from {maybe_path!r}')
                for path in detected_paths:
                    paths.append(path)

        self.__json_paths = paths

    def run(self):
        import app_logging
        app_logging.set_log_level(app_logging.INFO)
        import_jsons(*self.__json_paths)
        print('Done!')
