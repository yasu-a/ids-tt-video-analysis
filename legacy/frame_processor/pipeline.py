from .. import extract
from .common import check_entry_iterable


class Pipeline:
    def __init__(self, vfr: extract.VideoFrameReader, stages):
        self.__vfr = vfr
        self.__stages = stages

    def __getitem__(self, s: slice):
        entry_it = self.__vfr.iter_frame_entries(s)
        for stage in self.__stages:
            if not check_entry_iterable(stage):
                raise TypeError('not a stage', stage)
            entry_it = stage(entry_it)
        for entry in entry_it:
            yield entry.as_result()


def create_pipeline(stages):
    def pipeline(vfr: extract.VideoFrameReader):
        return Pipeline(vfr, stages)

    return pipeline
