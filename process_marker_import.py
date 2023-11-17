import argparse

import labels.marker
import process


class ProcessStageMarkerImport(process.ProcessStage):
    NAME = 'marker-import'
    ALIASES = 'mi',

    @classmethod
    def customize_parser(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument('json_paths', type=str, nargs='+')

    def __init__(self, json_paths: list[str]):
        self.__json_paths = json_paths

    def run(self):
        for json_path in self.__json_paths:
            print(f'Loading {json_path!r} ...', end=' ')
            _, marker_json_path = labels.marker.import_json(json_path)
            if marker_json_path:
                print('imported')
            else:
                print('already exists')
        print('Done!')
