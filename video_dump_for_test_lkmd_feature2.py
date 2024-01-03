import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

import async_writer
import npstorage_context as snp_context
import storage
import storage.npstorage as snp
import train_input

snp_context.just_run_registration()

storage.context.forbid_writing = True

if __name__ == '__main__':
    video_name = '20230205_04_Narumoto_Harimoto'

    rect = train_input.frame_rects.normalized(video_name)

    snp_vf = storage.create_instance(
        domain='numpy_storage',
        entity=video_name,
        context='frames',
        mode='r',
    )

    df = pd.read_csv(f'./features_out/{video_name}.csv')

    start, end = 1720, 1989
    target_idx_array = np.arange(start, end)

    col_x_src = np.array([f'{ch}_{x}_{y}' for x in range(16) for y in range(16) for ch in 'xy'])
    col_y_src = np.array(['ready', 'stay', 'play'])

    label_to_color = dict(
        ready=(192, 0, 0),
        stay=(0, 192, 0),
        play=(0, 0, 192)
    )

    with snp_vf as snp_vf, async_writer.AsyncVideoFrameWriter('out.mp4', fps=7) as vw:
        assert isinstance(snp_vf, snp.NumpyStorage)
        fis = snp_vf.get_array('fi', fill_nan=-1)
        target_mask = np.in1d(fis, target_idx_array)
        target_fis = fis[target_mask]
        target_entry_idx = np.arange(len(fis))[target_mask]
        for fi, ei in zip(target_fis, tqdm(target_entry_idx)):
            entry = snp_vf.get_entry(ei)
            assert isinstance(entry, snp_context.SNPEntryVideoFrame)
            img = entry.original.copy()
            row = df.loc[df['fi'] == fi, :]
            row_x = row[col_x_src].to_numpy().reshape(16, 16, 2)
            row_y = row[col_y_src]
            if isinstance(rect, train_input.RectNormalized):
                rect = rect.to_actual_scaled(width=img.shape[1], height=img.shape[0])
            img = img[rect.index3d]
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            img = cv2.resize(img, None, fx=4, fy=4)
            hw = np.array(img.shape[:2])
            for xi in range(16):
                for yi in range(16):
                    dx, dy = row_x[xi, yi]
                    if np.isnan(dx) or np.isnan(dy):
                        continue
                    pt1 = np.array([xi, yi]) / 16 * hw
                    pt2 = pt1 + np.array([dx, dy]) * 200
                    label = row_y.columns[row_y.astype(bool).iloc[0]][0]
                    cv2.arrowedLine(
                        img,
                        pt1.astype(int),
                        pt2.astype(int),
                        label_to_color[label],
                        thickness=3,
                        tipLength=0.3
                    )
            cv2.putText(img, f'[{label[0].upper()}] {fi}fr {entry.timestamp:.3f}s', (0, 60),
                        cv2.FONT_HERSHEY_PLAIN, 2, (192, 192, 192), thickness=2)
            vw.write(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
