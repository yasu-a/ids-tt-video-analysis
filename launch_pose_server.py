if __name__ == '__main__':
    from pose_server import PoseDetectorSingleProcessServer, PoseDetectorMultiProcessServer
    from util_pose import MoveNetPoseDetector

    MULTIPROCESS = False

    if MULTIPROCESS:
        server = PoseDetectorMultiProcessServer(
            port=13579,
            detector_class=MoveNetPoseDetector,
            n_unused_cpus=1
        )
        try:
            server.serve_forever()
        except InterruptedError:
            print('SHUTTING SERVER DOWN...')
            server.shutdown()
    else:
        server = PoseDetectorSingleProcessServer(
            port=13579,
            detector_class=MoveNetPoseDetector,
        )
        server.serve_forever()
