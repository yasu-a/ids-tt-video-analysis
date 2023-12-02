from matplotlib import pyplot as plt

import app_logging
from primitive_motion_detector import PMDetectorSource, PMDetectorResult


class PMDetectorTester:
    def __init__(self, source: PMDetectorSource, result: PMDetectorResult):
        self._src = source
        self._result = result
        self._logger = app_logging.create_logger(f'{__name__}#Tester')

    def test_all(self):
        self._logger.info('test_all')

        for field_name in dir(self):
            if not field_name.startswith('test_'):
                continue
            if field_name == 'test_all':
                continue
            getattr(self, field_name)()

    def test_detect_keypoints(self):
        self._logger.info('test_detect_keypoints')

        plt.figure(figsize=(10, 5))
        for i in range(2):
            plt.subplot(120 + i + 1)
            plt.imshow(self._result.original_images_clipped[i])
            plt.scatter(
                *self._result.keypoints[i].T[::-1],
                c='yellow',
                marker='x',
                s=500,
                linewidths=3
            )
            plt.axis('off')
        plt.suptitle('test_detect_keypoints')
        plt.tight_layout()
        plt.show()

    def test_matches(self):
        self._logger.info('test_matches')

        matches_tuple = {tuple(x) for x in self._result.match_index_pair.T}
        n_keys_a, n_keys_b = map(len, self._result.keyframes)
        fig, axes = plt.subplots(n_keys_a + 2, n_keys_b + 2, figsize=(40, 40))

        from tqdm import tqdm

        for i in tqdm(range(n_keys_a)):
            for j in range(n_keys_b):
                axes[i + 2, j + 2].bar([0], [self._result.distance_matrix[i, j]])
                axes[i + 2, j + 2].set_ylim(0, 1)
                if (i, j) in matches_tuple:
                    axes[i + 2, j + 2].scatter([0], [0.5], color='red', s=500)

        for i in range(n_keys_a):
            axes[i + 2, 0].imshow(self._result.original_images_clipped[0])
            axes[i + 2, 0].scatter(
                self._result.keypoints[0][i, 1],
                self._result.keypoints[0][i, 0],
                color='yellow',
                marker='x',
                s=200
            )
            axes[i + 2, 1].imshow(self._result.keyframes[0][i])
        for i in range(n_keys_b):
            axes[0, i + 2].imshow(self._result.original_images_clipped[1])
            axes[0, i + 2].scatter(
                self._result.keypoints[1][i, 1],
                self._result.keypoints[1][i, 0],
                color='yellow',
                marker='x',
                s=200
            )
            axes[1, i + 2].imshow(self._result.keyframes[1][i])

        for ax in axes.flatten():
            ax.axis('off')

        plt.suptitle('test_matches')
        fig.tight_layout()
        fig.show()

    def test_local_centroids(self):
        self._logger.info('test_local_centroids')

        plt.figure(figsize=(10, 5))

        for i in range(2):
            plt.subplot(1, 2, i + 1)
            plt.imshow(self._result.original_images_clipped[i])
            for j, mi in enumerate(self._result.match_index_pair[i]):
                plt.scatter(
                    *self._result.local_centroid[i][j][::-1],
                    color='yellow',
                    marker='x',
                    s=200
                )
            plt.axis('off')

        plt.suptitle('test_local_centroids')
        plt.tight_layout()
        plt.show()

    def test_global_centroids(self):
        self._logger.info('test_global_centroids')

        plt.figure(figsize=(10, 5))

        for i in range(2):
            plt.subplot(1, 2, i + 1)
            plt.imshow([self._src.target_frame.original_image,
                        self._src.next_frame.original_image][i])
            for j, mi in enumerate(self._result.match_index_pair[i]):
                plt.scatter(
                    *self._result.global_centroid[i][j][::-1],
                    color='yellow',
                    marker='x',
                    s=200
                )
            plt.axis('off')

        plt.suptitle('test_global_centroids')
        plt.tight_layout()
        plt.show()

    def test_local_centroids_normalized(self):
        self._logger.info('test_local_centroids_normalized')

        plt.figure(figsize=(10, 5))

        for i in range(2):
            plt.subplot(1, 2, i + 1)
            plt.imshow([self._src.target_frame.original_image,
                        self._src.next_frame.original_image][i])
            for j, mi in enumerate(self._result.match_index_pair[i]):
                xy = self._result.local_centroid_normalized[i][j][::-1]
                xy = xy * self._src.rect_actual_scaled.size + self._src.rect_actual_scaled.p_min
                plt.scatter(
                    *xy,
                    color='yellow',
                    marker='x',
                    s=200
                )
            # plt.axis('off')

        plt.suptitle('test_local_centroids_normalized')
        plt.tight_layout()
        plt.show()
