import pickle

import numpy as np
from tqdm import tqdm

import train_input

np.set_printoptions(suppress=True)

with dataset.VideoBaseFrameStorage(
        dataset.get_video_frame_dump_dir_path(),
        mode='r'
) as vf_storage:
    timestamp = vf_storage.get_all_of('timestamp')

train_input_df, rally_mask = train_input.load_rally_mask(
    'label_data/iDSTTVideoAnalysis_20230205_04_Narumoto_Harimoto.csv',
    timestamp
)


def split_vertically(n_split, offset, height, points):
    axis = 0
    n = height // n_split
    criteria = points[:, axis] - offset
    nth_split = criteria.astype(int) // n
    return tuple(
        np.where(np.minimum(nth_split, n_split - 1) == i)[0]
        for i in range(n_split)
    )


def create_rect():
    r = slice(70, 260), slice(180, 255)  # height, width

    # height: 奥の選手の頭から手前の選手の足がすっぽり入るように
    # width: ネットの部分の卓球台の幅に合うように

    def process_rect(rect):
        w = rect[1].stop - rect[1].start
        aw = int(w * 1.0)
        return slice(rect[0].start, rect[0].stop), slice(rect[1].start - aw, rect[1].stop + aw)

    r = process_rect(r)

    return r


rect = create_rect()

with dataset.MotionStorage(
        dataset.get_motion_dump_dir_path(),
        mode='r',
) as m_store:
    mv = []
    for i in tqdm(range(m_store.count())):
        data_dct = m_store.get(i)
        start, end = data_dct['start'], data_dct['end']
        start, end = start[~np.isnan(start[:, 0])], end[~np.isnan(end[:, 0])]
        assert len(start) == len(end), (start.shape, end.shape)
        mv.append(end - start)
    with open('out.pickle', 'wb') as f:
        pickle.dump(dict(mv=mv, timestamp=timestamp, rally_mask=rally_mask), f)
