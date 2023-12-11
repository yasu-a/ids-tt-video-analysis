from typing import NamedTuple

import numpy as np

import storage.npstorage as snp


class SNPEntryVideoFrame(NamedTuple):
    motion: np.ndarray
    original: np.ndarray
    timestamp: float
    fi: int  # frame-index, int


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
            },
            'fi': {
                'dtype': np.int32
            }
        },
        entry_named_tuple=SNPEntryVideoFrame
    )
)


class SNPEntryPrimitiveMotion(NamedTuple):
    start: np.ndarray
    end: np.ndarray
    timestamp: float
    fi: int


snp.register_storage_context(
    snp.StorageContext(
        name='primitive_motion',
        struct_definition={
            'start': {
                'dtype': np.float32
            },
            'end': {
                'dtype': np.float32
            },
            'timestamp': {
                'dtype': np.float32
            },
            'fi': {
                'dtype': np.int32
            }
        },
        entry_named_tuple=SNPEntryPrimitiveMotion
    )
)


class SNPEntryLKMotion(NamedTuple):
    start: np.ndarray
    velocity: np.ndarray  # normalized by rect and timestamps
    timestamp: float
    fi: int


snp.register_storage_context(
    snp.StorageContext(
        name='lk_motion',
        struct_definition={
            'start': {
                'dtype': np.float32
            },
            'velocity': {
                'dtype': np.float32
            },
            'timestamp': {
                'dtype': np.float32
            },
            'fi': {
                'dtype': np.int32
            }
        },
        entry_named_tuple=SNPEntryLKMotion
    )
)


def just_run_registration():
    pass
