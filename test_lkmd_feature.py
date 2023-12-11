import functools

import cv2
import numpy as np
from tqdm import tqdm

import async_writer
import npstorage_context as snp_context
import storage
import storage.npstorage as snp
import train_input

if __name__ == '__main__':
    video_name = '20230205_04_Narumoto_Harimoto'

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
        N = 16


        @functools.cache
        def tile_feature(i):
            entry: snp_context.SNPEntryLKMotion = snp_lk_motion.get_entry(i)

            mask = ~np.any(np.isnan(entry.start), axis=1)

            if entry.start is None:
                return None, None

            start = entry.start[mask, :]
            velocity = entry.velocity[mask, :]

            angle = np.arctan2(velocity[:, 1], velocity[:, 0])  # arctan2 receives (y, x)
            # print(velocity[:, 0].mean(), velocity[:, 1].mean(), angle.mean())
            length = np.linalg.norm(velocity, axis=1)

            idx = (start * N).astype(int)
            idx_flatten = idx[:, 0] * N + idx[:, 1]

            vel_ag_mean = np.zeros(shape=(N, N))
            vel_ln_mean = np.zeros(shape=(N, N))
            for xi in range(N):
                for yi in range(N):
                    a = angle[idx_flatten == xi * N + yi]
                    l = length[idx_flatten == xi * N + yi]
                    if a.size:
                        vel_ag_mean[xi, yi] = a.mean()
                        vel_ln_mean[xi, yi] = l.mean()
                    else:
                        vel_ag_mean[xi, yi] = np.nan

            vel_ag_mean.setflags(write=False)
            vel_ln_mean.setflags(write=False)

            return vel_ag_mean, vel_ln_mean


        def tile_feature_interpolate(i):
            assert i == int(i), i
            i = int(i)
            vel_ag_mean, vel_ln_mean = tile_feature(i)
            assert vel_ag_mean is not None
            return vel_ag_mean, vel_ln_mean


        def feature(i):
            f_ag, f_ln = [], []
            for j in range(i - 30 * 6, i + 30 * 6, 5):
                vel_ag_mean, vel_ln_mean = tile_feature_interpolate(j)
                f_ag.append(vel_ag_mean)
                f_ln.append(vel_ln_mean)
            f_ag, f_ln = np.stack(f_ag, axis=2), np.stack(f_ln, axis=2)
            return f_ag, f_ln


        print(feature(100)[0].shape)


        def show(i):
            frame_entry: snp_context.SNPEntryVideoFrame = snp_frames.get_entry(i)
            original = frame_entry.original.copy()

            vel_ag_mean, vel_ln_mean = tile_feature_interpolate(frame_entry.fi)

            S = 512

            original = original[train_input.frame_rects.actual_scaled(
                video_name,
                width=original.shape[1],
                height=original.shape[0]
            ).index3d]
            original = cv2.cvtColor(original, cv2.COLOR_RGB2BGR)
            img = cv2.resize(original, (S, S))
            for xi in range(N):
                for yi in range(N):
                    ag = vel_ag_mean[xi, yi]
                    if np.isnan(ag):
                        continue
                    ln = vel_ln_mean[xi, yi]
                    s = np.array([xi, yi]) / N * S + S / N / 2
                    d = np.array([np.cos(ag), np.sin(ag)]) * ln * (S * 0.2)

                    cv2.arrowedLine(
                        img,
                        tuple(map(int, s)),
                        tuple(map(int, s + d)),
                        (0, 255, 255),
                        thickness=2,
                        tipLength=0.3
                    )
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            vw.write(img)

            # plt.figure()
            # sns.heatmap(vel_ag_mean)
            # plt.show()
            # plt.close()
            # time.sleep(0.5)


        a = snp_lk_motion
        print()
        for i in tqdm(range(200, 400)):
            show(i)
