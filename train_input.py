import pandas as pd
import numpy as np


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


if __name__ == '__main__':
    df = load('./train/iDSTTVideoAnalysis_20230205_04_Narumoto_Harimoto.csv')
    print(df)
