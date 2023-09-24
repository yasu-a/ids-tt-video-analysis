import pandas as pd


def load(path):
    df = pd.read_csv(path)

    def time_mapper(s):
        minute, second = map(int, s.split(':'))
        return minute * 60 + second

    df = df.applymap(time_mapper)
    df = df.astype(float)

    return df


if __name__ == '__main__':
    df = load('./train/iDSTTVideoAnalysis_20230205_04_Narumoto_Harimoto.csv')
    print(df)
