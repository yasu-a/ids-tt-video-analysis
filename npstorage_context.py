from typing import NamedTuple

import numpy as np

import storage.npstorage as snp


class SNPEntryVideoFrame(NamedTuple):
    motion: np.ndarray
    original: np.ndarray
    timestamp: float


snp.register_storage_context(
    snp.StorageContext(
        name='frames',
        struct_definition={
            'motion': {
                'dtype': np.uint8
            },
            'original': {
                'dtype': np.uint8
            },
            'timestamp': {
                'dtype': np.float32
            }
        },
        entry_named_tuple=SNPEntryVideoFrame
    )
)


class SNPEntryLocalPeakMaxMotion(NamedTuple):
    start: float
    end: float
    frame_index: int
    timestamp: float


snp.register_storage_context(
    snp.StorageContext(
        name='lpm_motion',
        struct_definition={
            'start': {
                'dtype': np.float32
            },
            'end': {
                'dtype': np.float32
            },
            'frame_index': {
                'dtype': np.int32
            },
            'timestamp': {
                'dtype': np.float32
            }
        },
        entry_named_tuple=SNPEntryLocalPeakMaxMotion
    )
)


def just_run_registration():
    pass
