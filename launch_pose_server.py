if __name__ == '__main__':
    from pose_server import PoseDetectorServer
    from util_pose import MoveNetPoseDetector

    server = PoseDetectorServer(
        port=13579,
        detector_class=MoveNetPoseDetector,
        n_unused_cpus=1
    )
    try:
        server.serve_forever()
    except InterruptedError:
        print('SHUTTING SERVER DOWN...')
        server.shutdown()
