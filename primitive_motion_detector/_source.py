import numpy as np

from train_input import RectNormalized
from ._util import check_dtype_and_shape


class PMDetectorSourceTimeSeriesEntry:
    @property
    def original_image(self):
        return self.__original_image

    @property
    def diff_image(self):
        return self.__diff_image

    @property
    def timestamp(self):
        return self.__timestamp

    @property
    def height(self):
        # returns the frame height sampled from `self.frame_original`
        return self.original_image.shape[0]

    @property
    def width(self):
        # returns the frame width sampled from `self.frame_original`
        return self.original_image.shape[1]

    @property
    def frame_shape(self):
        return self.height, self.width, 3

    @property
    def _checker_for_frame_image(self):
        return check_dtype_and_shape(
            dtype=np.uint8,
            shape=self.frame_shape
        )

    def __init__(self, original_image, diff_image, timestamp):
        """
        PMDetectorの入力データのうちの時系列データ
        :param original_image: np.uint8 with shape(height, width, channels)
        :param diff_image: np.uint8 with shape(height, width, channels)
        :param timestamp: float
        """
        # assign members
        self.__original_image = original_image
        self.__diff_image = diff_image
        self.__timestamp = timestamp

        # check arguments
        self._checker_for_frame_image(original_image)
        self._checker_for_frame_image(diff_image)

        assert isinstance(timestamp, float), type(timestamp)


class PMDetectorSource:
    @property
    def target_frame(self):
        return self.__target_frame

    @property
    def next_frame(self):
        return self.__next_frame

    @property
    def rect_normalized(self):
        return self.__rect_normalized

    @property
    def rect_actual_scaled(self):
        return self.__rect_normalized.to_actual_scaled(
            width=self.width,
            height=self.height
        )

    @property
    def height(self):
        # returns the frame height sampled from `self.target_frame`
        return self.target_frame.height

    @property
    def width(self):
        # returns the frame width sampled from `self.target_frame`
        return self.target_frame.width

    @property
    def frame_shape(self):
        return self.target_frame.frame_shape

    def __init__(self, target_frame, next_frame, detection_rect_normalized):
        # noinspection GrazieInspection
        """
        PMDetectorの入力データ

        :param target_frame: PMDetectorInputTimeSeriesEntry
        :param next_frame: PMDetectorInputTimeSeriesEntry
        :param detection_rect_normalized: RectNormalized
        """
        # assign members
        self.__target_frame = target_frame
        self.__next_frame = next_frame
        self.__rect_normalized = detection_rect_normalized

        # check arguments
        assert isinstance(target_frame, PMDetectorSourceTimeSeriesEntry), target_frame
        assert isinstance(next_frame, PMDetectorSourceTimeSeriesEntry), next_frame
        assert target_frame.frame_shape == next_frame.frame_shape, \
            (target_frame.frame_shape, next_frame.frame_shape)
        assert isinstance(detection_rect_normalized, RectNormalized), detection_rect_normalized

    @property
    def frame_interval(self) -> float:
        return self.next_frame.timestamp - self.target_frame.timestamp
