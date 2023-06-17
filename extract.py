from tqdm import tqdm
import cv2
import os
import glob
import contextlib
import zipfile

path = os.path.expanduser(r'~\Desktop/idsttvideos/singles')
glob_pattern = os.path.join(path, r'**/*.mp4')
video_path = glob.glob(glob_pattern, recursive=True)[0]
print(video_path)

ZIP_DEST_BASE_PATH = r'./extract'

def generate_zip_path(video_path):
    _, video_name = os.path.split(video_path)
    name, _ = os.path.splitext(video_name)
    zip_name = f'{name}_extract.zip'
    zip_path = os.path.join(ZIP_DEST_BASE_PATH, name)
    return zip_path

# zip_path = generate_zip_path(video_path)
#
# with zipfile.ZipFile(zip_path, 'r') as zf:
#


@contextlib.contextmanager
def iter_frames(path, *, start=None, end=None, step=None):
    HEIGHT = 1080
    WIDTH = 1440

    cap = cv2.VideoCapture(path, apiPreference=cv2.CAP_ANY, params=[cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT, cv2.CAP_PROP_FRAME_WIDTH, WIDTH])

    if not cap.isOpened():
        raise ValueError("failed to open video")

    def it():
        i = start or 0
        fps = cap.get(cv2.CAP_PROP_FPS)
        spf = 1 / fps
        while True:
            if end is not None and i >= end:
                break
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ok, frame = cap.read()
            if not ok:
                break
            duration = spf * i
            yield duration, frame
            i += step or 1

    yield cap, it()

    cap.release()


def down_sample(frame, ratio):
    return cv2.resize(frame, None, None, ratio, ratio, cv2.INTER_LANCZOS4)


for rm_path in glob.iglob(r'./extract/*.png'):
    os.remove(rm_path)

start = 0 * 30
step = 20
end = 60 * 30

down_sampling_ratio = 0.5

with iter_frames(video_path, start=start, end=end, step=step) as (cap, frames):
    for duration, frame in tqdm(frames):
        # frame = down_sample(frame, down_sampling_ratio)
        name = f'{int(duration * 1000):08}.png'
        dst_path = os.path.join('./extract', name)
        cv2.imwrite(dst_path, frame)
