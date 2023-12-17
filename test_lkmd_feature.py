import collections
import functools
import os

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

import async_writer
import npstorage_context as snp_context
import storage
import storage.npstorage as snp
import train_input

storage.context.forbid_writing = True


def dump_motion_feature(video_name):
    vw = async_writer.AsyncVideoFrameWriter(
        path='out.mp4',
        fps=10
    )
    snp_lk_motion = storage.create_instance(
        domain='numpy_storage',
        entity=video_name,
        context='lk_motion',
        mode='r',
    )
    snp_frames = storage.create_instance(
        domain='numpy_storage',
        entity=video_name,
        context='frames',
        mode='r',
    )

    with vw as vw, snp_lk_motion as snp_lk_motion, snp_frames as snp_frames:
        assert isinstance(snp_lk_motion, snp.NumpyStorage)
        assert isinstance(snp_frames, snp.NumpyStorage)

        def extract_vector_complex(i):
            entry: snp_context.SNPEntryLKMotion = snp_lk_motion.get_entry(i)

            mask = ~np.any(np.isnan(entry.start), axis=1)

            if entry.start is None:
                return None, None

            start = entry.start[mask, :]
            velocity = entry.velocity[mask, :]

            angle = np.arctan2(velocity[:, 1], velocity[:, 0])  # arctan2 receives (y, x)
            length = np.linalg.norm(velocity, axis=1)

            z = length * np.exp(1j * angle)
            return start, z

        def bag_collector(z):
            assert z.ndim == 1, z.shape

            if len(z) <= 2:
                z_masked = z
            else:
                length = np.abs(z)
                mask = length >= np.percentile(length, 50)
                z_masked = z[mask]

            if len(z_masked) == 0:
                return np.nan

            return z_masked.mean()

        N_BANDS = 16

        @functools.cache
        def band_of_features(i):
            start, z = extract_vector_complex(i)

            if start is None:
                return None

            if start.size == 0:
                return np.full(N_BANDS, fill_value=np.nan)

            y = np.clip(start[:, 1], 0, 1 - 1e-6)  # TODO: investigate normalized values>=1
            y_idx = np.int32(y * N_BANDS)
            assert 0 <= y_idx.min() and y_idx.max() < N_BANDS, y_idx

            z_mean = np.array([bag_collector(z[y_idx == yi]) for yi in range(N_BANDS)])
            z_mean.setflags(write=False)

            return z_mean

        def generate_row_dct(i):
            z = band_of_features(i)  # z_mean on each band

            z_length = np.abs(z)
            z_angle = np.angle(z)
            z_x = z_length * np.cos(z_angle)
            z_y = z_length * np.sin(z_angle)

            z_length[np.isnan(z_length)] = 0
            z_x[np.isnan(z_x)] = 0
            z_y[np.isnan(z_y)] = 0

            dct = collections.OrderedDict()
            for i in range(N_BANDS):
                dct[f'norm_{i}'] = z_length[i]
            for i in range(N_BANDS):
                dct[f'arg_{i}'] = z_angle[i]
            for i in range(N_BANDS):
                dct[f'x_{i}'] = z_x[i]
            for i in range(N_BANDS):
                dct[f'y_{i}'] = z_y[i]

            return dct

        def generate_dataframe(indexes):
            rows = [generate_row_dct(i) for i in tqdm(indexes)]
            return pd.DataFrame(rows)

        ARROW_FACTOR = 60

        def show(ei):
            frame_entry: snp_context.SNPEntryVideoFrame = snp_frames.get_entry(ei)

            row = generate_row_dct(int(frame_entry.fi))

            img = frame_entry.original.copy()
            img = img[train_input.frame_rects.actual_scaled(
                video_name,
                width=img.shape[1],
                height=img.shape[0]
            ).index3d]
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            img = cv2.resize(img, None, fx=2, fy=2)
            h, w, _ = img.shape

            for i in range(N_BANDS):
                dx, dy = row[f'x_{i}'] * ARROW_FACTOR, row[f'y_{i}'] * ARROW_FACTOR
                start = w // 2, h // N_BANDS * i + h // N_BANDS // 2
                end = start[0] + int(dx), start[1] + int(dy)
                cv2.arrowedLine(
                    img,
                    start,
                    end,
                    (0, 255, 255),
                    thickness=2,
                    tipLength=0.3
                )

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            vw.write(img)

        # for i in tqdm(range(200, 300)):
        #     show(i)

        def to_csv(video_name):
            df_y = pd.read_csv(os.path.join('./label_data/grand_truth', video_name + '.csv'))
            df_x = generate_dataframe(df_y.index).astype(np.float16)
            df = df_x.join(df_y)
            print(df)
            df.to_csv(f'{video_name}.csv')

        to_csv(video_name)


if __name__ == '__main__':
    def main():
        print(storage.StoragePath.list_storages())


    main()
