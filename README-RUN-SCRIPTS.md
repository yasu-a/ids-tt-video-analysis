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

### ProcessStageVideoDump [process_video_dump.py](process_defs/process_video_dump.py)

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
  - `-s`/`--step`：このステップ数で飛ばし飛ばしフレームを書き出す（>=3？）
  - `-d`/`--diff-luminance-scale`/`--scale`：差分画像を何倍するか

### ProcessStageExtractRectCLI [process_extract_rect_cli.py](process_defs/process_extract_rect_cli.py)

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

### ProcessStageLabeledFrameDump [process_labeled_frame_dump.py](process_defs/process_labeled_frame_dump.py) <font color="red">現在開発中</font>

インポートしたラベルデータから【フレームデータ vs. ラベル】のデータセットを生成する。

```shell
python main.py labeled-frame-dump
```

```shell
python main.py lfd
```

- 位置引数
    - （不明）
- キーワード引数
    - （不明）

### ProcessStageMarkerImport [process_marker_import.py](process_defs/process_marker_import.py)

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

### ProcessStagePrimitiveMotionDump [process_primitive_motion_dump.py](process_defs/process_primitive_motion_dump.py)

次のデータから[util_extrema_feature_motion_detector.py](./util_extrema_feature_motion_detector.py)
によるモーションデータを生成する。

- `video-dump`のフレームダンプ
- `extract-rect-cli`の矩形データ

```shell
python main.py primitive-motion-dump
```

```shell
python main.py pmd
```

- 位置引数
  - `video_name`：動画の名前
- キーワード引数
  - （なし）

### ProcessStagePrimitiveMotionVisualize [process_primitive_motion_visualize.py](process_defs/process_primitive_motion_visualize.py)

次のデータからモーションベクトルを動画に重ねて可視化する。

- `video-dump`のフレームダンプ
- `primitive-motion-dump`のモーションダンプ

```shell
python main.py primitive-motion-visualize
```

```shell
python main.py pmv
```

- 位置引数
  - `video_name`：動画の名前
- キーワード引数
  - `--start`：開始フレーム
  - `--stop`：終了フレーム
  - `--out-path`：出力先