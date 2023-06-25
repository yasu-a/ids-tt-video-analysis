from extractimio import _dump_array, VideoFrameIO
import io
from PIL import Image

from tqdm import tqdm
import glob
import os
import timeit

import cv2
import numpy as np
import pandas as pd

from extractimio import VideoFrameIO, MultiprocessingVideoFrameIO

path = os.path.expanduser(r'~\Desktop/idsttvideos/singles')
glob_pattern = os.path.join(path, r'**/*.mp4')
video_path = glob.glob(glob_pattern, recursive=True)[0]
print(video_path)

cache_base_path = os.path.expanduser(r'~\Desktop/vio_cache')

def main():
    import imageio.v2 as iio

    STEP = 1

    with VideoFrameIO(video_path, cache_base_path=cache_base_path) as vio:
        frame_index, frame = list(vio.loc[:1])[0]
        