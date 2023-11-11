import multiprocessing as mp

import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

from util_pose_common import *


class PoseDetectorUtilsMixin:
    @staticmethod
    def crop_keeping_aspect(image: np.ndarray, max_size: int):
        size_x, size_y = image.shape
        resizing_ratio = max_size / max(size_x, size_y)
        dst_size_x, dst_size_y = int(size_x * resizing_ratio), int(size_y * resizing_ratio)
        assert 0 < dst_size_x <= max_size and 0 < dst_size_y <= max_size, (dst_size_x, dst_size_y)
        resized_image = cv2.resize(image, (dst_size_x, dst_size_y))
        return resized_image


class PoseDetector(PoseDetectorUtilsMixin):
    def detect(self, src_image: np.ndarray) -> PoseDetectionResult:
        raise NotImplementedError()


# https://qiita.com/michelle0915/items/f69e6255595fe82799b8
class MoveNetPoseDetector(PoseDetector):
    model = hub.load('https://tfhub.dev/google/movenet/multipose/lightning/1')
    model = model.signatures['serving_default']

    def detect(self, src_image: np.ndarray) -> PoseDetectionResult:
        # preprocess image
        img = cv2.cvtColor(src_image, cv2.COLOR_BGR2RGB)
        img = tf.cast(img, dtype=tf.int32)
        img = tf.expand_dims(img, axis=0)
        img = tf.image.resize_with_pad(img, 256, 256)
        img = tf.cast(img, dtype=tf.int32)

        # infer posing
        outputs = self.model(img)
        result = outputs['output_0'].numpy()  # shape = (1, n_bodies, 56)

        # parse model's output
        bodies = []
        for body_array in result[0]:
            part_ys = body_array[0::3][:17]
            part_xs = body_array[1::3][:17]
            part_scores = body_array[2::3][:17]
            part_centroids = np.concatenate([part_xs, part_ys]).T

            body_bbox = body_array[-5:][[1, 0, 3, 2]]
            body_score = body_array[-1]

            body = Body(
                bbox=body_bbox,
                score=body_score,
                part_centroids=part_centroids,
                part_scores=part_scores
            )
            bodies.append(body)

        detection_result = PoseDetectionResult(bodies=bodies)
        return detection_result


class PoseDetectorProcess:
    def __init__(self, *, detector: PoseDetector, process_index=None, max_q_size):
        self.__detector = detector
        self.__process_index = process_index
        self.__current_job_index = 0
        self.__q_input = mp.Queue(maxsize=max_q_size)
        self.__q_output = mp.Queue()
        self.__outputs = {}  # job_index -> result
        self.__process = mp.Process(
            target=self._worker,
            name=f'pose detector process #{process_index}',
            args=(),
            kwargs={}
        )
        print(f' PROCESS#{self.__process_index} created')

    def await_ready(self):
        assert self.__q_output.get() == 'READY'

    def start(self):
        print(f' PROCESS#{self.__process_index} started')
        self.__process.start()

    def __get_job_index(self):
        job_index = self.__current_job_index
        self.__current_job_index += 1
        return job_index

    def send_image(self, image: np.ndarray):
        job_index = self.__get_job_index()
        self.__q_input.put((job_index, image))
        return job_index

    def send_stop(self):
        self.__q_input.put(None)

    def join(self):
        print(f' PROCESS#{self.__process_index} joined')
        self.__process.join()

    def close(self):
        print(f' PROCESS#{self.__process_index} closed')
        self.__process.close()

    def _worker(self):
        print(f' PROCESS#{self.__process_index} worker entered')
        self.__q_output.put('READY')
        while True:
            # if self.__q_input.empty():
            #     time.sleep(0.02)
            #     continue

            input_raw = self.__q_input.get()
            if input_raw is None:  # stop signal
                break

            job_index, image = input_raw
            result = self.__detector.detect(image)
            print(f' PROCESS#{self.__process_index} finish detecting job #{job_index}')
            self.__q_output.put((job_index, result))
        print(f' PROCESS#{self.__process_index} worker left')

    def __receive_output(self):
        while not self.__q_output.empty():
            job_index, result = self.__q_output.get(block=False)
            self.__outputs[job_index] = result

    def retrieve_result(self, job_index):
        self.__receive_output()
        result = self.__outputs.get(job_index)
        return result
