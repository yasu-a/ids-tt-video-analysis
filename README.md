# ids-tt-video-analysis

Tチーム動画分析のリポジトリです。

# 現時点でのプロセス

1. 動画からフレームを抽出
    - はじめは`decord`でやっていた
    - `cv2.VideoCapture`でのフレーム抽出に変更
        - `cv2.VideoCapture`でも`cap.grab`でフレームを読み飛ばすことで高速化できることが判明
    - (a)...抽出したRGBフレーム画像
2. フレーム画像(a)どうしの差分抽出
    - (b)...(a)どうしの差分フレーム（=(a)の１階微分）
    - フレームの差分とその次のフレームの差分の幾何平均によりノイズを取り除けることが判明
    - (c)...差分(b)どうしの幾何平均（≒(a)の２階微分）
3. モーション抽出
    - ノイズ処理後の差分(c)の輝度極大点をキーポイントとして使用（SIFTなどのキーポイントに相当）
    - (d)...(c)から抽出したキーポイント周辺の(a)におけるキー局所領域（SIFTなどのディスクリプタに相当）
    - 隣り合う2つの(c)によるキー領域(d)を全数比較し、両方向からマッチしたキーポイントの差分をモーションとして抽出
    - (e)...抽出したモーション
4. 特徴量抽出
    - (f1)...モーション(e)のx軸方向成分の平均を特徴量にする
    - (f2)...特徴量を時系列分割し、互いの分割の差を生成し、特徴量追加
5. 機械学習
    - (m1)...特徴量(f2)をランダムフォレストで学習＆推定
    - (m2)...特徴量(f1)をLSTMで学習＆推定
    - 予測の良さ（まだ定量化できていない）：(m2)>(m1)

![img](presen_materials/slides/flow.png)

![img](presen_materials/note_rnn_rally_detection/rally_detection_rnn.gif)

# スクリプトの実行

## アプリ内の環境パスの編集

アプリ内環境パスは[./dataset_config.json](./dataset_config.json)に記述します。

データ構造：

```json
{
  "env-config": {
    "<デバイス名1>": {
      "項目1": "値",
      "項目2": "値",
      "項目3": "値",
      "...": "..."
    },
    "<デバイス名2>": {
      "...": "..."
    },
    "<デバイス名3>": {
      "...": "..."
    },
    "...": {
      "...": "..."
    }
  }
}
```

### 必須項目

- `data-location`：ダンプデータやキャッシュデータを保存・参照する場所
- `debug-data-location`:デバッグ時に参照されることがある`data-location`を汚染しないための別の場所
- `video-location`：動画を参照する場所
- `default-video-name`：デバッグ時に参照されることがある動画の名前

### 特殊項目・ディレクティブ

- `.comment`：このキーに対応する値はコメントとして無視される
- `.inherit`：このキーに対応するデバイス名の構成を継承する

## コマンドラインの実行

このフォルダを`cd`としてコマンドを実行する。

```shell
python main.py <subcommand> <args> ...
```

### ProcessStageVideoDump [process_video_dump.py](./process_video_dump.py)

動画のフレームのオリジナル画像・差分画像をダンプする。

```shell
python main.py video-dump
```

```shell
python main.py vd
```

- 位置引数
    - `video_name`：動画の名前
- キーワード引数
    - `-r`/`--resize-ratio`：リサイズ比率（0.0～1.0）
    - `-s`/`--step`：このステップ数で飛ばし飛ばしフレームを書き出（>=3？）
    - `-d`/`--diff-luminance-scale`/`--scale`：差分画像を何倍するか

### ProcessStageExtractRectCLI [process_extract_rect_cli.py](./process_extract_rect_cli.py)

動画のモーション抽出を行う矩形領域を設定するための対話プログラムを起動する。

```shell
python main.py extract-rect-cli
```

```shell
python main.py cli-rect
```

- 位置引数
    - `video_name`：動画の名前
- キーワード引数
    - （なし）

### ProcessStageMarkerImport [process_labeled_frame_dump.py](./process_labeled_frame_dump.py) <font color="red">現在開発中</font>

インポートしたラベルデータから【フレームデータ vs. ラベル】のデータセットを生成する。

```shell
python main.py labeled-frame-dump
python main.py lfd
```

- 位置引数
    - （不明）
- キーワード引数
    - （不明）

### ProcessStageMarkerImport [process_marker_import.py](./process_marker_import.py)

[GitHub:ids-tt-video-marker](https://github.com/yasu-a/ids-tt-video-marker)による動画のラベルデータをインポートする。

```shell
python main.py marker-import
```

```shell
python main.py mi
```

- 位置引数
    - `json_paths`：jsonパス（複数，globパターン可）
- キーワード引数
    - （なし）

## 実装と処理結果の紹介

### Local-Maxによるキーポイント検出とモーション検出

- [util_extrema_feature_motion_detector.py](./util_extrema_feature_motion_detector.py)
- [note_extrema_key_frame_motion_detection.py](notes/note_extrema_key_frame_motion_detection.py)
- [note_local_max_featured_motion_detection_mp4_dump.py](notes/note_local_max_featured_motion_detection_mp4_dump.py)

![img](presen_materials/local_max_feature_motion_vectors.gif)

Key-frame distance matrix
![img](presen_materials/local_max_feature_dist_mat.png)

### Motion Centroid Correction

- [note_keyframe_center_correction.py](notes/note_keyframe_center_correction.py)

![img](presen_materials/motion_centroid_correction/compare.png)

#### Correction Disabled

![img](presen_materials/motion_centroid_correction/out_without_motion_correction.gif)

#### Correction Enabled

手前の選手の頭に注目

Disabledではベクトルが暴れているがenabledでは暴れが抑えられている

![img](presen_materials/motion_centroid_correction/out_with_motion_correction.gif)

#### Correction Disabled

![img](presen_materials/motion_centroid_correction/out_without_motion_correction.png)

#### Correction Enabled

![img](presen_materials/motion_centroid_correction/out_with_motion_correction.png)

### Random Forest によるモーションの分類

https://github.com/yasu-a/ids-tt-video-analysis/blob/master/presen_materials/start_detection.mp4

### LSTM によるモーションの分類

https://github.com/yasu-a/ids-tt-video-analysis/blob/master/presen_materials/note_rnn_rally_detection/rally_detection_rnn.mp4

![img](presen_materials/note_rnn_rally_detection/rally_detection_rnn.png)
