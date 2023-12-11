import argparse


class ProcessStage:
    NAME = None
    ALIASES = None
    ENABLED = True

    @classmethod
    def customize_parser(cls, parser: argparse.ArgumentParser) -> None:
        raise NotImplementedError()
    def run(self):
        raise NotImplementedError()
