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
        s.connect(self.__address)
        data_pickle = pickle.dumps(images)
        s.sendall(data_pickle)
        data_pickle = _socket_receive_all(s)
        return pickle.loads(data_pickle)
