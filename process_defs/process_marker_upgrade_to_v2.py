import argparse
import codecs
import glob
import json
import os.path
from pprint import pformat

from frozendict import frozendict

import app_logging
import process

logger = app_logging.create_logger('__name__')


class CommandError(RuntimeError):
    pass


class ProcessStageMarkerUpgradeToV2(process.ProcessStage):
    NAME = 'marker-upgrade-to-v2'
    ALIASES = 'mu2',

    @classmethod
    def customize_parser(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument('json_paths', type=str, nargs='+')
        parser.add_argument('-d', '--author-hash-digest', type=str, default=None)
        parser.add_argument('-n', '--author-node', type=str, default=None)
        parser.add_argument('-s', '--author-system', type=str, default=None)
        parser.add_argument('-x', '--dont-write-back', action='store_true')

    def __init__(
            self,
            json_paths: list[str],
            author_hash_digest,
            author_node,
            author_system,
            dont_write_back
    ):
        # パスをパースしてフラットなjson_pathのリストに変換する
        paths = []
        for maybe_path in json_paths:
            if os.path.exists(maybe_path):
                paths.append(maybe_path)
            else:
                detected_paths = glob.glob(maybe_path, recursive=True)
                if not detected_paths:
                    logger.warning(f'Warning: nothing extracted from {maybe_path!r}')
                for path in detected_paths:
                    paths.append(path)
        self.__json_paths = paths

        # その他の定数
        self.__write_back = not dont_write_back

        # author_elementをひとますストア
        self.__author_element = {
            "hash-digest": author_hash_digest,
            "node": author_node,
            "system": author_system
        }

    def _retrieve_author_element(self):
        author_elements = set()
        for path in self.__json_paths:
            with codecs.open(path, 'r', 'utf-8') as f:
                json_root = json.load(f)

            try:
                version = json_root['meta']['json-version']
            except KeyError:
                continue

            assert version == 2, f'version != 2 not allowed: {version=}'

            author_element = frozendict(json_root['meta']['author'])

            # if set(author_element.keys()) != {'hash-digest', 'node', 'system'}:
            #     continue

            author_elements.add(author_element)

        if len(author_elements) == 0:
            raise CommandError('No author element found')
        elif len(author_elements) == 1:
            return dict(author_elements.pop())
        else:
            logger.warning('Multiple author elements found:')
            for author_element in author_elements:
                logger.warning(dict(author_element))
            raise CommandError('Multiple author elements found')

    def upgrade_to_v2(self, source_json_path, source_json_root):
        logger.info(f' *** Upgrading json to v2: {source_json_path!r}')

        import video_marker_json_compat as compat

        upgraded_json_root = compat.convert(
            json_path=source_json_path,
            json_root=source_json_root,
            version_to=2
        )
        upgraded_json_root['meta']['author'] = self.__author_element

        if self.__write_back:
            with codecs.open(source_json_path, 'w', 'utf-8') as f:
                json.dump(upgraded_json_root, f, indent=2, sort_keys=True, ensure_ascii=False)
            logger.info(f'Write back to {source_json_path!r}')
        else:
            logger.info('Write back not performed!!!')
            logger.info('Meta:')
            logger.info(pformat(upgraded_json_root['meta']))
        logger.info(f'Done')

    def run(self):
        try:
            # もしauthor_elementに欠損があったら
            if any(v is None for v in self.__author_element.values()):
                # jsonを回ってauthor_elementをかき集める
                retrieved_author_element = self._retrieve_author_element()
                # 集めたauthor_elementで欠損を補填
                for k in self.__author_element.keys():
                    if self.__author_element[k] is not None:
                        if self.__author_element[k] != retrieved_author_element[k]:
                            # 欠損値でなくとも一致性を確認する
                            logger.warning('Retrieved author element mismatch')
                            logger.warning('Commandline:')
                            logger.warning(self.__author_element)
                            logger.warning('Retrieved:')
                            logger.warning(retrieved_author_element)
                            raise CommandError('Retrieved author element mismatch')
                    else:
                        # 欠損値は更新する
                        self.__author_element[k] = retrieved_author_element[k]

            logger.info('author_element:')
            logger.info(self.__author_element)

            for source_json_path in self.__json_paths:
                with codecs.open(source_json_path, 'r', 'utf-8') as f:
                    source_json_root = json.load(f)
                    self.upgrade_to_v2(source_json_path, source_json_root)
        except CommandError as e:
            logger.error(f'CommandError: {e.args}')
