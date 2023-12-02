from primitive_motion_detector import PMDetectorParameter, PMDetectorSource, PMComputer


class PMDetector:
    def __init__(self, parameter: PMDetectorParameter):
        self.__parameter = parameter

    def compute(self, source: PMDetectorSource):
        return PMComputer(self.__parameter, source).compute()
