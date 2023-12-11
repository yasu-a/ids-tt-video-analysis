import argparse

from ._stage import ProcessStage

_process_stage_registration = {}


def register_process_in_module(module):
    for key, value in vars(module).items():
        # filter value
        if not isinstance(value, type):
            continue
        if not issubclass(value, ProcessStage):
            continue

        # class `ProcessStage` found
        process_stage_type = value

        # filter by `ENABLED` flag
        if not process_stage_type.ENABLED:
            continue

        # register class to `_process_stage_registration`
        global _process_stage_registration

        # register class by canonical name
        # _process_stage_registration[process_stage_type.NAME] = process_stage_type

        # register class by its alias
        for alias in process_stage_type.ALIASES or []:
            _process_stage_registration[alias] = process_stage_type


def _handler(process_stage_type):
    def _handler_impl(args):
        kwargs = vars(args)
        kwargs.pop('_handler')
        # for k in list(kwargs.keys()):
        #     if kwargs.get(k) is None:
        #         kwargs.pop(k)
        process_stage = process_stage_type(**kwargs)
        process_stage.run()

    return _handler_impl


def _global_parser():
    parser = argparse.ArgumentParser()
    sub_parsers = parser.add_subparsers()

    for stage_name, process_stage_type in _process_stage_registration.items():
        sub_parser = sub_parsers.add_parser(stage_name)
        process_stage_type.customize_parser(sub_parser)
        sub_parser.set_defaults(_handler=_handler(process_stage_type))

    return parser


def run(argv=None):
    parser = _global_parser()
    args = parser.parse_args(argv)
    if hasattr(args, '_handler'):
        # noinspection PyProtectedMember
        args._handler(args)
    else:
        parser.print_help()
