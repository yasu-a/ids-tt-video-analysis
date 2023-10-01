import json
import os
import pickle
import traceback
import warnings

import numpy as np

import config


class MemoryMapStorage:
    def generate_config(self):
        return {
            '__status__': {
                'dtype': np.uint8
            }
        }

    STATUS_UNFINISHED = 0
    STATUS_FINISHED = 1

    def __init__(self, dir_path, mode, max_entries=None):
        if mode == 'w':
            assert max_entries is not None, max_entries
        elif mode == 'r':
            assert max_entries is None, max_entries
        else:
            assert False, mode

        os.makedirs(dir_path, exist_ok=True)

        self.__dir_path = dir_path
        self.__mode = mode
        self.__max_entries = max_entries

        self.__data = {}
        self.__info_json = {
            'shape': {}
        }

        if mode == 'r':
            self.__load_json()

        self.__initialized = False

        self._config = self.generate_config()

    def __initialize(self):
        if self.__mode == 'w':
            self.__get_array('__status__', shape=())
            self.__get_array('__status__')[:] = self.STATUS_UNFINISHED

    def __ensure_initialized(self):
        if not self.__initialized:
            self.__initialize()
            self.__initialized = True

    @property
    def json_path(self):
        return os.path.join(self.__dir_path, 'info.json')

    def __load_json(self):
        if os.path.exists(self.json_path):
            with open(self.json_path, 'r') as f:
                self.__info_json = json.load(f)

    def __dump_json(self):
        with open(self.json_path, 'w') as f:
            json.dump(self.__info_json, f, indent=2, sort_keys=True)

    def __memmap_file_path(self, name):
        return os.path.join(self.__dir_path, name + '.mmap')

    def __get_array(self, name, shape=None):
        if name not in self.__data:
            if self.__mode == 'w':
                shape = self.max_entry_count(), *shape
                assert all(x is not None for x in shape), shape
                self.__data[name] = np.memmap(
                    filename=self.__memmap_file_path(name),
                    mode='w+',
                    dtype=self._config[name]['dtype'],
                    shape=shape
                )
                self.__info_json['shape'][name] = shape
                self.__dump_json()
            elif self.__mode == 'r':
                shape = self.__info_json['shape'][name]
                self.__data[name] = np.memmap(
                    filename=self.__memmap_file_path(name),
                    mode='r+',
                    dtype=self._config[name]['dtype'],
                ).reshape(shape)
            else:
                assert False, self.__mode

        return self.__data[name]

    def mark_finished(self, i):
        self.__get_array('__status__')[i] = self.STATUS_FINISHED

    def max_entry_count(self):
        if self.__max_entries is not None:
            return self.__max_entries
        elif '__status__' in self.__data:
            return self.__get_array('__status__').shape[0]
        else:
            assert 'max_entries not given', self.__max_entries

    def finished_entry_count(self):
        return int(np.sum(self.__get_array('__status__') == self.STATUS_FINISHED))

    def count(self):
        return self.finished_entry_count()

    def put(self, i, data_dct):
        assert self.__mode == 'w', self.__mode
        assert set(data_dct.keys()) == set(self._config.keys() - {'__status__'}), \
            (data_dct.keys(), self._config.keys() - {'__status__'})
        self.__ensure_initialized()
        for name in data_dct.keys():
            shape = getattr(data_dct[name], 'shape', None) or tuple()
            a = self.__get_array(name, shape=shape)
            a[i, ...] = data_dct[name]
        self.mark_finished(i)

    def get(self, i):  # FIXME: __status__ filtering
        assert self.__mode == 'r', self.__mode
        self.__ensure_initialized()
        data_dct = {}
        for name in self._config.keys():
            a = self.__get_array(name)
            data_dct[name] = a[i, ...]
        return data_dct

    def get_all_of(self, name):
        assert self.__mode == 'r', self.__mode
        self.__ensure_initialized()
        return np.array(self.__get_array(name))

    def close(self):
        for k, v in self.__data.items():
            print(f'[{type(self).__name__}] closing mmap {k!r}...')
            # noinspection PyProtectedMember
            v._mmap.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


class VideoFrameStorage(MemoryMapStorage):
    def generate_config(self):
        return super().generate_config() | {
            'motion': {
                'dtype': np.uint8
            },
            'original': {
                'dtype': np.uint8
            },
            'timestamp': {
                'dtype': np.float32
            }
        }


class MotionStorage(MemoryMapStorage):
    def generate_config(self):
        return super().generate_config() | {
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
        }


class FrameDumpIO:
    def __init__(self, video_name=None):
        self.__video_name = coerce_video_name(video_name)

    def get_storage(self, mode, max_entries=None):
        return VideoFrameStorage(
            os.path.join(
                config.FRAME_DUMP_DIR_PATH,
                self.__video_name,
            ),
            mode,
            max_entries=max_entries
        )

    @property
    def video_path(self):
        return os.path.join(
            config.VIDEO_DIR_PATH,
            f'{self.__video_name}.mp4'
        )


def _print_warning(msg):
    warnings.warn(msg)
    traceback.print_stack()


def coerce_video_name(video_name):
    if video_name is None:
        video_name = config.DEFAULT_VIDEO_NAME
        _print_warning('Default video name used')
    return video_name


def get_video_path(video_name=None):
    video_name = coerce_video_name(video_name)
    return os.path.join(
        config.VIDEO_DIR_PATH,
        video_name + '.mp4'
    )


def get_video_frame_dump_dir_path(video_name=None):
    video_name = coerce_video_name(video_name)
    return os.path.join(
        config.FRAME_DUMP_DIR_PATH,
        video_name,
    )


def get_motion_dump_dir_path(video_name=None):
    video_name = coerce_video_name(video_name)
    return os.path.join(
        config.MOTION_DUMP_DIR_PATH,
        video_name
    )


class PickleCache:
    def __init__(self, video_name):
        self.__root_path = os.path.join(
            config.FEATURE_CACHE_PATH,
            video_name
        )
        os.makedirs(self.__root_path, exist_ok=True)

    def dump(self, name, obj):
        path = os.path.join(self.__root_path, name + '.pickle')
        with open(path, 'wb') as f:
            pickle.dump(obj, f)

    def load(self, name):
        path = os.path.join(self.__root_path, name + '.pickle')
        try:
            with open(path, 'rb') as f:
                return pickle.load(f)
        except (pickle.PickleError, FileNotFoundError):
            return None
