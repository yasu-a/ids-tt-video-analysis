import argparse


class ProcessStage:
    NAME = None

    @classmethod
    def customize_parser(cls, parser: argparse.ArgumentParser) -> None:
        raise NotImplementedError()

    def run(self):
        raise NotImplementedError()


_process_stage_registration = {}


def register_process_in_module(module):
    for key, value in vars(module).items():
        if not isinstance(value, type):
            continue
        if not issubclass(value, ProcessStage):
            continue
        process_stage_type = value
        global _process_stage_registration
        _process_stage_registration[process_stage_type.NAME] = process_stage_type


def _handler(process_stage_type):
    def _handler_impl(args):
        kwargs = vars(args)
        kwargs.pop('_handler')
        for k in list(kwargs.keys()):
            if kwargs.get(k) is None:
                kwargs.pop(k)
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
