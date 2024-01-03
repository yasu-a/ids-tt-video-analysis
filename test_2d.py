from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd


def extract_df_from_zipfile(zip_path, csv_name, df_mapper):
    import zipfile
    import pandas as pd

    with zipfile.ZipFile(zip_path, 'r') as zf:
        with zf.open(csv_name, 'r') as f:
            df = pd.read_csv(f)
            df = df_mapper(df, csv_name)
    return df


def retrieve_zip_path():
    import glob
    zip_path = glob.glob('./idsttva_dataset_2d_*.zip')[0]
    return zip_path


def mp_extract_dfs(zip_path, df_mapper) -> list['pd.DataFrame']:
    # import time
    import zipfile
    from tqdm import tqdm
    from concurrent.futures import ProcessPoolExecutor as PoolExecutor
    from concurrent.futures import as_completed

    def dispatch():
        with PoolExecutor(max_workers=None) as pool:
            with zipfile.ZipFile(zip_path, 'r') as zf:
                futures = []
                for csv_name in zf.namelist():
                    future = pool.submit(extract_df_from_zipfile, zip_path, csv_name, df_mapper)
                    futures.append(future)
            for future in tqdm(as_completed(futures), total=len(futures), desc='Extracting CSVs'):
                df = future.result()
                yield df

    # ts = time.perf_counter()
    dfs = list(dispatch())
    # te = time.perf_counter()
    # print(te - ts)

    return dfs


def load_df(zip_path, df_mapper, df_collector):
    dfs = mp_extract_dfs(zip_path=zip_path, df_mapper=df_mapper)
    df = df_collector(dfs)
    return df


def memory_info():
    import os
    import psutil
    print('[memory_info(MiB)]', psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2)


# =======================================
# TRAIN
# =======================================

def df_mapper_train(df, csv_name):
    df = df.drop(df[df['invalid'] == 1].index)
    df = df.drop(columns=['Unnamed: 0'])
    df['csv_name'] = csv_name
    return df


def df_collector_train(dfs):
    import pandas as pd
    df = pd.concat(dfs, axis='rows').reset_index()
    return df


def main():
    import pandas as pd
    import numpy as np

    pd.set_option('display.max_columns', 500)
    pd.set_option('display.max_rows', 20)
    pd.set_option('display.width', 200)
    np.set_printoptions(edgeitems=7, linewidth=200)

    df = load_df(
        zip_path=retrieve_zip_path(),
        df_mapper=df_mapper_train,
        df_collector=df_collector_train
    )
    df.replace(-np.inf, 0, inplace=True)  # TODO: investigate where -inf comes from

    col_x_src = np.array([f'{axis}_{x}_{y}' for x in range(16) for y in range(16) for axis in 'xy'])
    col_y_src = np.array(['ready', 'stay', 'play'])

    mat_x = df[col_x_src].to_numpy()
    df_y = df[col_y_src]
    a_label = col_y_src[df[col_y_src].to_numpy().argmax(axis=1)]

    print(mat_x)
    print(df_y)
    print(a_label)

    x, y = mat_x[0::2], mat_x[1::2]

    memory_info()


if __name__ == '__main__':
    main()
