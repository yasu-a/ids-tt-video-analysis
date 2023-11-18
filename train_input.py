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
    with open('label_data/rect.json', 'r') as f:
        json_root = json.load(f)
    rect_lst = json_root.get(video_name)
    rect_lst = slice(rect_lst[0], rect_lst[1]), slice(rect_lst[2], rect_lst[3])
    return rect_lst


def update_rect(video_name, rect):
    # rect_lst = rect[0].start, rect[0].stop, rect[1].start, rect[1].stop
    rect_lst = [rect[0][0], rect[0][1], rect[1][0], rect[1][1]]
    with open('label_data/rect.json', 'r') as f:
        json_root = json.load(f)
    json_root[video_name] = rect_lst
    with open('label_data/rect.json', 'w') as f:
        json.dump(json_root, f, indent=2, sort_keys=True)
