from typing import NamedTuple

import numpy as np

__all__ = 'Body', 'PoseDetectionResult'


class Body(NamedTuple):
    bbox: np.ndarray  # float[4(x_min, y_min, x_max, y_max)]
    score: np.ndarray  # float
    part_centroids: np.ndarray  # float[N_PARTS, 2(x, y)]
    part_scores: np.ndarray  # float[N_PARTS]

    @property
    def n_parts(self):
        return len(self.part_bboxes)


class PoseDetectionResult(NamedTuple):
    bodies: list[Body]

    @property
    def n_bodies(self):
        return self.bodies
