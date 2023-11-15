import argparse
import collections
import sys

import numpy as np

import app_logging
import npstorage_context
import storage
import storage.npstorage as snp

npstorage_context.just_run_registration()

app_logging.set_log_level(app_logging.WARN)

"""
inspect context
inspect struct frames
inspect meta frames
inspect value[s] frames 10
inspect value[s] frames --at 10
inspect value[s] frames 10 [20 [5]]
inspect value[s] frames  --start 10 [--stop 20 [--step 5]] [--key motion]
inspect none/null/check/check_null frames
"""

_operation_parser = {}


class OperationBase:
    def __init__(self, cl: list[str]):
        self._cl = cl

    def parser(self) -> argparse.ArgumentParser:
        raise NotImplementedError()

    def parse(self):
        return self.parser().parse_args(self._cl)

    def run(self):
        raise NotImplementedError()


def register_operation_parser(op: str, aliases: list[str] = None):
    aliases = aliases or []

    def decorator(cls):
        global _operation_parser
        for op_candidate in op, *aliases:
            _operation_parser[op_candidate] = cls

    return decorator


def find_operation_parser(op: str) -> 'type[OperationBase]':
    return _operation_parser.get(op)


@register_operation_parser(op='context')
class OperationContext(OperationBase):
    def parser(self):
        parser = argparse.ArgumentParser(
            description='Inspect dumps of storage.npstorage'
        )
        parser.add_argument('-s', '--size', '-f', '--files', action='store_true')
        return parser

    def run(self):
        args = self.parse()

        dct = collections.defaultdict(lambda: collections.defaultdict(list))
        for sp in storage.StoragePath.list_storages():
            dct[sp.domain][sp.entity].append(sp)
        for domain, domain_dct in dct.items():
            print(f'[Domain] {domain!r}')
            for entity, sp_lst in domain_dct.items():
                print(f' [Entity] {entity!r}')
                for sp in sp_lst:
                    print(f'  [Context] {sp.context!r}')
                    print(f'   {sp.path} {sp.total_size // 1000:20,d} KB')
                    if args.size:
                        for path, size in sp.list_sizes():
                            print(f'   - .{path[len(sp.path):]:40s} {size // 1000:20,d} KB')


def index_validater(value):
    split = value.split(':')

    if len(split) == 1:
        return [int(split[0])]

    while len(split) < 3:
        split += ['']
    return slice(*(None if s == '' else int(s) for s in split))


@register_operation_parser(op='content')
class OperationContext(OperationBase):
    def parser(self):
        parser = argparse.ArgumentParser(
            description='Inspect dumps of storage.npstorage'
        )
        parser.add_argument('entity', type=str)
        parser.add_argument('context', type=str)
        parser.add_argument('index', type=index_validater)
        parser.add_argument('-k', '--key', type=str, required=False, nargs=1)
        parser.add_argument('-f', '--full', action='store_true')
        parser.add_argument('--imshow', action='store_true')
        return parser

    def run(self):
        args = self.parse()

        with storage.create_instance(
                domain='numpy_storage',
                entity=args.entity,
                context=args.context,
                mode='r'
        ) as st:
            assert isinstance(st, snp.NumpyStorage)
            index = np.arange(st.count())[args.index]
            for i in index:
                print(f'[Index] {i}')
                entry = st.get_entry(i)
                # noinspection PyProtectedMember
                for name, value in entry._asdict().items():
                    if args.key and name not in args.key:
                        continue
                    if value is None:
                        print(f' [Array] {name!r}')
                        print('NULL')
                    else:
                        print(f' [Array] {name!r} {value.shape} / {value.size} / {value.dtype}')
                        if args.full:
                            print(value)
                        elif args.imshow:
                            import cv2
                            print('cv2.imshow()')
                            cv2.imshow('cv2.imshow', cv2.cvtColor(value, cv2.COLOR_BGR2RGB))
                            cv2.waitKey()
                        else:
                            if value.size < 32:
                                print(value)
                            else:
                                print('(--full to show full data)')


@register_operation_parser(op='check', aliases=['null', 'check-null', 'none', 'info'])
class OperationContext(OperationBase):
    def parser(self):
        parser = argparse.ArgumentParser(
            description='Inspect dumps of storage.npstorage'
        )
        parser.add_argument('entity', type=str)
        parser.add_argument('context', type=str)
        return parser

    def run(self):
        args = self.parse()

        with storage.create_instance(
                domain='numpy_storage',
                entity=args.entity,
                context=args.context,
                mode='r'
        ) as st:
            assert isinstance(st, snp.NumpyStorage)
            print(f'Count: {st.count()}')
            status = {}
            for name in st.get_array_names():
                column_status = st.get_status(name)
                if np.any(column_status == snp.STATUS_INVALID):
                    print(
                        f'Unfilled [Array] {name!r} '
                        f'{(column_status == snp.STATUS_INVALID).sum()}/{st.count()}'
                    )
                status[name] = column_status
            for i in range(st.count()):
                row_status = np.array([status[name][i] for name in st.get_array_names()])
                consistent = np.all(np.diff(row_status) == 0)
                if consistent:
                    continue
                print(f'Inconsistency [Index] {i}')
                # noinspection PyProtectedMember
                for name, value in st.get_entry(i)._asdict().items():
                    if value is None:
                        print(f'  [Array] {name:<20s} NULL')
                    else:
                        print(f'  [Array] {name:<20s} {value.shape} / {value.size} / {value.dtype}')


def run(cl: list[str]):
    cl = cl or ['context']
    op, *rest = cl
    op_parser_type = find_operation_parser(op)
    if op_parser_type is None:
        print(
            f'Invalid operation! Available operations are {list(_operation_parser.keys())}.',
            file=sys.stderr
        )
        return
    op_parser = op_parser_type(rest)
    op_parser.run()


print(snp.list_storage_context())

# run(['check', '20230205_04_Narumoto_Harimoto', 'frames'])

if __name__ == '__main__':
    run(sys.argv[1:])
