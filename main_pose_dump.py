from tqdm import tqdm

import pose

dataset.forbid_writing()

KEYPOINT_THRESHOLD = 0.2
POSE_SERVER_PORT = 13579


def main():
    # TODO: replace dataset to npstorage
    with dataset.VideoBaseFrameStorage(
            dataset.get_video_frame_dump_dir_path(video_name=None, high_res=False),
            mode='r',
    ) as vf_store:
        def iter_batches():
            it = iter(tqdm(range(vf_store.count())))
            while True:
                data_batch = []
                for _ in range(8):
                    i = next(it)
                    fr = vf_store.get(i)
                    data_batch.append(fr)
                if not data_batch:
                    break
                yield data_batch

        for data_batch in iter_batches():
            frames = [dct['original'] for dct in data_batch]
            timestamps = [dct['timestamp'] for dct in data_batch]
            client = pose.PoseDetectorClient(port=POSE_SERVER_PORT)
            results = client.detect(frames)

            # TODO: continue implementation


#
# def render(image, keypoints_list, scores_list, bbox_list):
#     render = image.copy()
#     for i, (keypoints, scores, bbox) in enumerate(zip(keypoints_list, scores_list, bbox_list)):
#         if bbox[4] < 0.2:
#             continue
#
#         cv2.rectangle(render, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
#
#         # 0:nose, 1:left eye, 2:right eye, 3:left ear, 4:right ear, 5:left shoulder,
#         # 6:right shoulder, 7:left elbow, 8:right elbow, 9:left wrist, 10:right wrist,
#         # 11:left hip, 12:right hip, 13:left knee, 14:right knee, 15:left ankle, 16:right ankle
#         # 接続するキーポイントの組
#         kp_links = [
#             (0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (0, 6), (5, 6), (5, 7), (7, 9), (6, 8),
#             (8, 10), (11, 12), (5, 11), (11, 13), (13, 15), (6, 12), (12, 14), (14, 16)
#         ]
#         for kp_idx_1, kp_idx_2 in kp_links:
#             kp_1 = keypoints[kp_idx_1]
#             kp_2 = keypoints[kp_idx_2]
#             score_1 = scores[kp_idx_1]
#             score_2 = scores[kp_idx_2]
#             if score_1 > KEYPOINT_THRESHOLD and score_2 > KEYPOINT_THRESHOLD:
#                 cv2.line(render, tuple(kp_1), tuple(kp_2), (0, 0, 255), 2)
#
#         for idx, (keypoint, score) in enumerate(zip(keypoints, scores)):
#             if score > KEYPOINT_THRESHOLD:
#                 cv2.circle(render, tuple(keypoint), 4, (0, 0, 255), -1)
#
#     return render


if __name__ == '__main__':
    main()
