# ids-tt-video-analysis

![img](presen_materials/note_rnn_rally_detection/rally_detection_rnn.gif)

# Run Scripts

1. この階層に`config.py`という名前のファイルを作る
2. 以下の定数を書き込む

|     VARIABLE NAME      |                                   DESCRIPTION                                    |                        EXAMPLE                         |
|:----------------------:|:--------------------------------------------------------------------------------:|:------------------------------------------------------:|
|    `VIDEO_DIR_PATH`    |               Path to the directory containing the original video                | `os.path.expanduser(r'~/Desktop/idsttvideos/singles')` |
| `FRAME_DUMP_DIR_PATH`  |  Dump destination of [main_process_video_dump.py](./main_process_video_dump.py)  |        `r'H:\idstt\iDSTTVideoFrameDump\frames'`        |
| `MOTION_DUMP_DIR_PATH` | Dump destination of [main_process_motion_dump.py](./main_process_motion_dump.py) |       `r'H:\idstt\iDSTTVideoFrameDump\motions'`        |
|  `DEFAULT_VIDEO_NAME`  |  Default video name to be used in each process when no video names is specified  |           `'20230205_04_Narumoto_Harimoto'`            |

# Scripts

|                             NAME                             |      DESCRIPTION       |
|:------------------------------------------------------------:|:----------------------:|
|                  [dataset.py](./dataset.py)                  |    `np.memmap`のI/O     |
|  [main_process_video_dump.py](./main_process_video_dump.py)  |  動画からフレームと差分フレームを生成する  |
| [main_process_motion_dump.py](./main_process_motion_dump.py) | 差分フレームからモーションベクトルを生成する |
|                       [train](./train)                       |         教師データ          |
|              [train_input.py](./train_input.py)              |        教師データI/O        |

# Local-Max Featured Motion Detection

- [util_extrema_feature_motion_detector.py](./util_extrema_feature_motion_detector.py)
- [note_extrema_key_frame_motion_detection.py](./note_extrema_key_frame_motion_detection.py)
- [note_local_max_featured_motion_detection_mp4_dump.py](./note_local_max_featured_motion_detection_mp4_dump.py)

![img](presen_materials/local_max_feature_motion_vectors.gif)

Key-frame distance matrix
![img](presen_materials/local_max_feature_dist_mat.png)

# Motion Centroid Correction

- [note_keyframe_center_correction.py](./note_keyframe_center_correction.py)

![img](presen_materials/motion_centroid_correction/compare.png)

Disabled

![img](presen_materials/motion_centroid_correction/out_without_motion_correction.gif)

Enabled

![img](presen_materials/motion_centroid_correction/out_with_motion_correction.gif)

Disabled

![img](presen_materials/motion_centroid_correction/out_without_motion_correction.png)

Enabled

![img](presen_materials/motion_centroid_correction/out_with_motion_correction.png)

# Motion Energy Random Forest Rally-Start Classification

https://github.com/yasu-a/ids-tt-video-analysis/blob/master/presen_materials/start_detection.mp4

# Partial Motion Mean Full Frame Differential Mean LSTM Rally Classification

https://github.com/yasu-a/ids-tt-video-analysis/blob/master/presen_materials/note_rnn_rally_detection/rally_detection_rnn.mp4

![img](presen_materials/note_rnn_rally_detection/rally_detection_rnn.png)
