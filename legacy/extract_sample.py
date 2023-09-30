# デスクトップの./idsttvideos/singlesの中にある動画から一つを選び
# その先頭1000フレームまで`out.mp4`に毎`STEP`フレームおきに書き出すサンプル

# 標準ライブラリ
import glob
import os

# imageio：動画の書き出しに使う
import imageio as iio
# tqdm：ループの進捗を表示してくれる
from tqdm import tqdm

# extractモジュール
from extract import VideoFrameReader

# 動画を見つける
path = os.path.expanduser(r'~\Desktop/idsttvideos/singles')
glob_pattern = os.path.join(path, r'**/*.mp4')
video_path = glob.glob(glob_pattern, recursive=True)[0]

# video_pathに動画のパスが入る
print(video_path)

# `STEP`の定義
STEP = 5

# 動画のパスからVideoFrameReaderインスタンスを生成
vfr = VideoFrameReader(video_path)

# 書き出し先の作成
out = iio.get_writer(
    '../out.mp4',
    format='FFMPEG',
    fps=vfr.fps / STEP,
    mode='I',
    codec='h264',
)

# フレームを切り出す
for frame_time, frame_array in tqdm(vfr[:1000:STEP]):
    # それぞれのフレームを書き込む
    out.append_data(frame_array)

# 書き出し先を閉じる
out.close()
