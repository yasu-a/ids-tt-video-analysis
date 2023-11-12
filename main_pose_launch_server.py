if __name__ == '__main__':
    from pose.server import PoseDetectorSingleProcessServer
    from pose.detector import MoveNetPoseDetector

    import app_logging

    logger = app_logging.create_logger(__name__)

    MULTIPROCESS = False

    if MULTIPROCESS:
        raise NotImplementedError()
        # server = PoseDetectorMultiProcessServer(
        #     port=13579,
        #     detector_class=MoveNetPoseDetector,
        #     n_unused_cpus=1
        # )
        # try:
        #     server.serve_forever()
        # except InterruptedError:
        #     print('SHUTTING SERVER DOWN...')
        #     server.shutdown()
    else:
        logger.info('Main start')
        server = PoseDetectorSingleProcessServer(
            port=13579,
            detector_class=MoveNetPoseDetector,
        )
        server.serve_forever()
        logger.info('Main end')