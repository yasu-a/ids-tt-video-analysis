import argparse

import numpy as np

import app_logging
import lk_motion_detector as lk
import npstorage_context as snp_context
import process
import storage
import storage.npstorage as snp
import train_input

logger = app_logging.create_logger(__name__)


def _pad_xys_f32(mat, max_n):
    if mat is None:
        mat = []
    mat = np.array(mat)
    if mat.shape == (0,):
        return np.full((max_n, 2), np.nan)
    assert mat.ndim == 2 and mat.shape[1] == 2, mat.shape
    mat = mat.astype(np.float32)
    n_pad = max_n - mat.shape[0]
    if n_pad < 0:
        raise ValueError(f'array is full; {max_n=} is too small: {mat.shape}')
    pad = np.full((n_pad, 2), np.nan)
    mat = np.concatenate([mat, pad], axis=0)
    assert len(mat) == max_n, mat.shape
    return mat


class ProcessStageLKMotionDump(process.ProcessStage):
    NAME = 'lk-motion-dump'
    ALIASES = 'lkmd',

    @classmethod
    def customize_parser(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument('video_name', type=str)
        parser.add_argument('-n', '--max-points-per-frame', '--max-points', type=int, default=4096)
        parser.add_argument('--lk-win-size', type=int, default=15),
        parser.add_argument('--lk-max-level', type=int, default=2),
        parser.add_argument('--lk-criteria-eps', type=float, default=0.03)
        parser.add_argument('--lk-criteria-count', type=int, default=10)
        parser.add_argument('--f-max-corners', type=int, default=256)
        parser.add_argument('--f-quality-level', type=float, default=0.1)
        parser.add_argument('--f-min-distance', type=int, default=20)
        parser.add_argument('--f-block-size', type=int, default=20)
        # no effect, see FIXME in lk_motion_detector
        # parser.add_argument('-r', '--original-resizing-scale', type=float, default=0.3)
        parser.add_argument('-t', '--max-track-length-seconds', type=float, default=0.4)
        parser.add_argument('-p', '--min-producible-track-length-seconds', type=float, default=0.1)
        parser.add_argument('-v', '--min-valid-track-length-seconds', type=float, default=0.05)
        parser.add_argument('-d', '--detect-interval-frames', type=int, default=1)
        parser.add_argument('-m', '--min-velocity-full-normalized', type=float, default=0.05)
        parser.add_argument('-i', '--imshow', action='store_true')
        parser.add_argument('-o', '--video-dump-path', type=str, default='')

    def __init__(
            self,
            video_name: str,
            max_points_per_frame,
            lk_win_size,
            lk_max_level,
            lk_criteria_eps,
            lk_criteria_count,
            f_max_corners,
            f_quality_level,
            f_min_distance,
            f_block_size,
            # original_resizing_scale,
            max_track_length_seconds,
            min_producible_track_length_seconds,
            min_valid_track_length_seconds,
            detect_interval_frames,
            min_velocity_full_normalized,
            imshow,
            video_dump_path,
            original_resizing_scale=0.3
    ):
        self.__video_name = video_name
        logger.info(f'{video_name=}')

        self.__max_points_per_frame = max_points_per_frame
        logger.info(f'{max_points_per_frame=}')

        import cv2

        self.__parameter = lk.LKMotionDetectorParameter(
            rect=train_input.frame_rects.normalized(video_name),
            lk_params=dict(
                winSize=(lk_win_size, lk_win_size),
                maxLevel=lk_max_level,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                          lk_criteria_count,
                          lk_criteria_eps)
            ),
            feature_params=dict(
                maxCorners=f_max_corners,
                qualityLevel=f_quality_level,
                minDistance=f_min_distance,
                blockSize=f_block_size
            ),
            original_resizing_scale=original_resizing_scale,
            max_track_length_seconds=max_track_length_seconds,
            min_producible_track_length_seconds=min_producible_track_length_seconds,
            min_valid_track_length_seconds=min_valid_track_length_seconds,
            detect_interval_frames=detect_interval_frames,
            min_velocity_full_normalized=min_velocity_full_normalized,
            imshow=imshow,
            video_dump_path=video_dump_path
        )

    def run(self):
        frame_producer = lk.FrameProducer.from_video_name(self.__video_name)
        frame_count = frame_producer.meta.frame_count

        detector = lk.LKMotionDetector(self.__parameter)
        computer = detector.computer(frame_producer)

        with storage.create_instance(
                domain='numpy_storage',
                entity=self.__video_name,
                context='lk_motion',
                mode='w',
                n_entries=frame_count
        ) as snp_lk_motion:
            assert isinstance(snp_lk_motion, snp.NumpyStorage)

            for frame, tracks in computer.iter_results():
                assert isinstance(frame, lk.LKDetectorFrame), frame
                assert isinstance(tracks, lk.Tracks), tracks

                start_arr = tracks.keypoint(
                    invariance='rect',
                    rect=computer.actual_scaled_rect
                )
                velocity_arr = tracks.velocity(
                    reliability='producible',
                    invariance='both',
                    rect=computer.actual_scaled_rect
                )
                flag_arr = tracks.producible()

                entry_start = []
                entry_velocity = []
                for start, velocity, producible in zip(start_arr, velocity_arr, flag_arr):
                    if not producible:
                        continue
                    entry_start.append(start)
                    entry_velocity.append(velocity)
                entry_start = _pad_xys_f32(entry_start, self.__max_points_per_frame)
                entry_velocity = _pad_xys_f32(entry_velocity, self.__max_points_per_frame)

                # df = pd.DataFrame(
                #     np.concatenate([entry_start, entry_velocity], axis=1),
                #     columns=['start_x', 'start_y', 'velocity_x', 'velocity_y']
                # )
                # print(df)

                entry = snp_context.SNPEntryLKMotion(
                    start=entry_start,
                    velocity=entry_velocity,
                    timestamp=frame.timestamp,
                    fi=frame.frame_index
                )
                snp_lk_motion[frame.frame_index] = entry
