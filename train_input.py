import json

import numpy as np
import pandas as pd


def load(path):
    df = pd.read_csv(path)

    def time_mapper(s):
        minute, second = map(int, s.split(':'))
        return minute * 60 + second

    df = df.applymap(time_mapper)
    df = df.astype(float)

    return df


def load_rally_mask(path, timestamps):
    train_input_df = load(path)

    s, e = train_input_df.start.to_numpy(), train_input_df.end.to_numpy()
    r = np.logical_and(s <= timestamps[:, None], timestamps[:, None] <= e).sum(axis=1)
    r = r > 0

    rally_mask = r.astype(np.uint8)

    return train_input_df, rally_mask


def load_rect(video_name=None):
    video_name = dataset.coerce_video_name(video_name)
    with open('./train/rect.json', 'r') as f:
        json_root = json.load(f)
    rect_lst = json_root.get(video_name)
    rect_lst = slice(rect_lst[0], rect_lst[1]), slice(rect_lst[2], rect_lst[3])
    return rect_lst


def update_rect(video_name, rect):
    video_name = dataset.coerce_video_name(video_name)
    rect_lst = rect[0].start, rect[0].stop, rect[1].start, rect[1].stop
    with open('./train/rect.json', 'r') as f:
        json_root = json.load(f)
    json_root[video_name] = rect_lst
    with open('./train/rect.json', 'w') as f:
        json.dump(json_root, f, indent=2, sort_keys=True)


if __name__ == '__main__':
    df = load('./train/iDSTTVideoAnalysis_20230205_04_Narumoto_Harimoto.csv')
    print(df)
