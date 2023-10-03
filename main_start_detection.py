import numpy as np

np.set_printoptions(suppress=True)

import train_input

from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(suppress=True)

from feature_generator import FeatureGenerator

Y_DATA_MARGIN = 10


def create_y_data(rally_mask):
    rm = rally_mask.astype(int)
    mask = (rm[1:] - rm[:-1]) > 0
    index_rally_begin = np.where(mask)[0]

    assert rm.ndim == 1, rm.shape
    index_delta = np.arange(rm.size)[:, None] - index_rally_begin
    nearest_rally_begin_index = index_rally_begin[np.abs(index_delta).argmin(axis=1)]
    nearest_rally_begin_index_delta = nearest_rally_begin_index - np.arange(rm.size)
    # y = (-Y_DATA_MARGIN < nearest_rally_begin_index_delta) & (
    #             nearest_rally_begin_index_delta < 0)
    y = (0 < nearest_rally_begin_index_delta) & (
            nearest_rally_begin_index_delta < Y_DATA_MARGIN)
    # y = np.abs(nearest_rally_begin_index_delta) < Y_DATA_MARGIN // 2

    # fig, axes = plt.subplots(3, 1, figsize=(100, 8))
    # axes[0].plot(rm[:-1])
    # axes[1].plot(nearest_rally_begin_index_delta)
    # axes[2].plot(y)
    # fig.show()

    return y


if __name__ == '__main__':
    fg = FeatureGenerator('20230205_04_Narumoto_Harimoto')
    N = 3000
    values = []
    for i, f in zip(range(N), fg.iter_motion_vector_diff_feature()):
        values.append(f)
    values = np.stack(values)
    print(values.shape)

    label = train_input.load_rally_mask(
        f'./train/iDSTTVideoAnalysis_{fg.video_name}.csv',
        timestamps=fg.timestamp
    )[1].astype(bool)[:N]

    N_PLOT = 5, 5
    fig, axes = plt.subplots(*N_PLOT, figsize=(7 * N_PLOT[0], 3 * N_PLOT[1]))
    axes = axes.flatten()
    for ax, f_row in tqdm(zip(axes, range(values.shape[1]))):
        x, y = values[:, f_row], label
        x_pos, x_neg = x[y], x[~y]
        # ax.scatter(x, y)
        ax.boxplot([x_pos, x_neg], labels=['pos', 'neg'], vert=False)
        ax.set_title(str(f_row))
    plt.tight_layout()
    plt.show()

    # video_names = [
    #     '20230205_04_Narumoto_Harimoto',
    #     '20230219_03_Narumoto_Ito',
    #     '20230225_02_Matsushima_Ando'
    # ]
    # 
    # a = FeatureGenerator('20230205_04_Narumoto_Harimoto').create(label=True)
    # x, y = a['feature'], create_y_data(a['label']).astype(bool)
    # y_src = y.copy()
    # idx = np.arange(y.size)
    # sample_idx = np.concatenate([
    #     np.random.choice(idx[~y], y.sum(), replace=True),
    #     idx[y]
    # ])
    # x, y = x[sample_idx], y[sample_idx]
    # print(y.sum(), (~y).sum())
    # 
    # from sklearn.ensemble import RandomForestClassifier
    # 
    # cl = RandomForestClassifier(
    #     max_depth=3,
    #     n_estimators=256,
    #     verbose=True,
    #     random_state=42,
    #     n_jobs=4
    # )
    # 
    # cl.fit(x, y)
    # 
    # vn_test = '20230219_03_Narumoto_Ito'
    # a = FeatureGenerator(vn_test).create(label=False)
    # x = a['feature']
    # 
    # y_pred = cl.predict(x)
    # 
    # fig, axes = plt.subplots(2, 1, figsize=(100, 4), sharex=True)
    # axes[0].imshow(np.tile(y_pred[:, None], 30).T)
    # axes[1].imshow(np.tile(y_src[:, None], 30).T)
    # plt.tight_layout()
    # plt.show()
    # 
    # fps = 1 / np.diff(a['timestamp']).mean()
    # print(f'{fps=}')
    # 
    # from PIL import Image
    # 
    # with dataset.VideoFrameStorage(
    #         dataset.get_video_frame_dump_dir_path(vn_test),
    #         mode='r'
    # ) as vf_store:
    #     images = []
    # 
    #     STEP = 3
    #     FAST = 2
    #     for i in tqdm(range(0, int(a['timestamp'].size * 0.3), STEP)):
    #         img = vf_store.get(i)['original']
    #         if y_pred[i]:
    #             img[:, :30:, :] = [0, 255, 0]
    #         else:
    #             img[:, :30:, :] = [0, 0, 255]
    #         images.append(Image.fromarray(img))
    # 
    #     images[0].save('output.gif',
    #                    save_all=True, append_images=images[1:], optimize=False,
    #                    duration=1 / fps * STEP / FAST,
    #                    loop=0)
