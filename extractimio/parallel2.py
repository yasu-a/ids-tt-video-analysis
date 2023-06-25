import datetime
import collections
from typing import NamedTuple
import glob
import os
import pickle
import time
import zipfile

import cv2
import imageio.v2 as iio
import numpy as np

import gzip

import multiprocessing as mp

from .common import Locator
from .diskio import VideoFrameCache, dump_array, load_array
from .single import VideoFrameIO


class QueryQueue:
    def __init__(self):
        pass


class FetchWorker:
    def __init__(self):
        pass


class FrameCache:
    def __init__(self):
        pass


class QueryWorker:
    def __init__(self):
        pass


class MultiprocessingVideoIO2:
    def __init__(self):
        pass
