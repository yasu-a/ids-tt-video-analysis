import gzip
import pickle
import zipfile

import numpy as np


def dump_array(f, a):
    dtype = a.dtype
    shape = a.shape
    raw = a.flatten().tobytes()
    obj = dict(
        dtype=dtype,
        shape=shape,
        raw=gzip.compress(raw)
    )
    pickle.dump(obj, f)


def load_array(f):
    obj = pickle.load(f)
    dtype = obj['dtype']
    shape = obj['shape']
    raw = gzip.decompress(obj['raw'])
    return np.frombuffer(raw, dtype=dtype).reshape(shape)


class VideoFrameCache:
    def __init__(self, cache_path):
        self.__cache_path = cache_path
        self.__zf: zipfile.ZipFile | None = None

    def open(self):
        if self.__zf is not None:
            raise ValueError()
        self.__zf = zipfile.ZipFile(
            self.__cache_path,
            mode='a',
            compression=zipfile.ZIP_STORED
        )

    def close(self):
        if self.__zf is None:
            raise ValueError()
        self.__zf.close()
        self.__zf = None

    def __encode_name(self, i):
        return str(i)

    def __decode_name(self, name):
        i = int(name)
        return i

    # TODO: cache, careful to modification
    def __get_name_by_index(self, i):
        target_name = self.__encode_name(i)
        for info in self.__zf.infolist():
            if info.filename == target_name:
                return info.filename
        return None

    def get_frame(self, i):
        name = self.__get_name_by_index(i)
        if name is None:
            return None
        with self.__zf.open(name, 'r') as f:
            try:
                return load_array(f)
            except EOFError:
                return None

    def put_frame(self, i, frame):
        if frame is None:
            raise ValueError()
        name = self.__encode_name(i)
        with self.__zf.open(name, 'w') as f:
            dump_array(f, frame)
