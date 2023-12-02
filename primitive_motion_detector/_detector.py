from ._computer import PMComputer
from ._parameter import PMDetectorParameter
from ._source import PMDetectorSource


class PMDetector:
    def __init__(self, parameter: PMDetectorParameter):
        self.__parameter = parameter

    def computer(self, source: PMDetectorSource):
        return PMComputer(self.__parameter, source)

    def compute(self, source: PMDetectorSource):
        return self.computer(source).compute()
