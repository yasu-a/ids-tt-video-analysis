import argparse


class ProcessStage:
    NAME = None
    ALIASES = None

    @classmethod
    def customize_parser(cls, parser: argparse.ArgumentParser) -> None:
        raise NotImplementedError()

    def run(self):
        raise NotImplementedError()
