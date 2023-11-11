import pickle
import socket
import traceback

from pose_common import *
from util_pose import *


class PoseDetectorSingleProcessServer:
    def __init__(self, *, host='localhost', port, detector_class: type[PoseDetector]):
        self.__address = host, port
        self.__detector = detector_class()

    def serve_forever(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        print(self.__address)
        s.bind(self.__address)
        s.listen(32)
        try:
            while True:
                connection, client_address = s.accept()
                print('ESTABLISHED', connection, client_address)

                try:
                    data_pickle = recv_blob(connection)
                except ConnectionError:
                    traceback.print_exc()
                    continue

                print(' received', len(data_pickle) // 1000, 'KBytes')

                obj = pickle.loads(data_pickle)
                if isinstance(obj, list):
                    input_images = obj
                else:
                    input_images = [obj]
                print(' received', len(input_images), 'images')

                results = [self.__detector.detect(image) for image in input_images]

                data_pickle = pickle.dumps(results)
                send_blob(connection, data_pickle)
                connection.close()
        finally:
            s.close()


class PoseDetectorMultiProcessServer:
    def __init__(self, *, host='localhost', port, detector_class: type[PoseDetector],
                 n_unused_cpus=1):
        self.__address = host, port

        self.__n_processes = max(1, mp.cpu_count() - n_unused_cpus)
        self.__processes = [
            PoseDetectorProcess(
                detector=detector_class(),
                process_index=process_index,
                max_q_size=16
            ) for process_index in range(self.__n_processes)]

    def shutdown(self):
        for p in self.__processes:
            p.send_stop()
        for p in self.__processes:
            p.join()
        for p in self.__processes:
            p.close()

    def serve_forever(self):
        for p in self.__processes:
            p.start()

        for p in self.__processes:
            p.await_ready()

        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        print(self.__address)
        s.bind(self.__address)
        s.listen(32)
        while True:
            connection, client_address = s.accept()
            print('ESTABLISHED', connection, client_address)

            data_pickle = _socket_receive_all(connection)
            print(' received', len(data_pickle) // 1000, 'KBytes')

            obj = pickle.loads(data_pickle)
            if isinstance(obj, list):
                input_images = obj
            else:
                input_images = [obj]
            print(' received', len(input_images), 'images')

            # assign processes for each image
            process_assignment = [
                i % self.__n_processes
                for i in range(len(self.__processes))
            ]
            job_indexes = [
                self.__processes[process_index].send_image(image)
                for image, process_index in zip(input_images, process_assignment)
            ]
            results = [None] * len(input_images)
            while len(results) < len(input_images):
                for i in range(len(input_images)):
                    if results[i] is not None:
                        continue
                    results[i] \
                        = self.__processes[process_assignment[i]].retrieve_result(job_indexes[i])

            data_pickle = pickle.dumps(results)
            connection.sendall(data_pickle)
            connection.close()