import pickle
import socket

import numpy as np

from pose_common import *
from util_pose_common import *


class PoseDetectorClient:
    def __init__(self, *, host='localhost', port):
        self.__address = host, port

    def detect(self, images: list[np.ndarray]) -> list[PoseDetectionResult]:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            s.connect(self.__address)
            data_pickle = pickle.dumps(images)
            send_blob(s, data_pickle)
            data_pickle = recv_blob(s)
        finally:
            s.close()
        return pickle.loads(data_pickle)
