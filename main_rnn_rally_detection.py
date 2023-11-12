import matplotlib.pyplot as plt

import numpy as np

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


SEQ_LEN = 50


def x_rnn_reshape(x):
    # x: npa[# of samples, # of features]
    slide = np.lib.stride_tricks.sliding_window_view(x, window_shape=(SEQ_LEN, 1))[..., 0]
    batch = np.swapaxes(slide, 1, 2)
    return batch


def y_rnn_reshape(y):
    # y2 = create_y_data(y)
    y = y.astype(np.int8)

    n_outer, n_inner = 8, 2

    diff = np.concatenate([[0], y[1:] - y[:-1]])
    idx = np.arange(y.size)
    rise_idx, fall_idx = idx[diff == 1], idx[diff == -1]
    rise_dist, fall_dist = idx - rise_idx[:, None], idx - fall_idx[:, None]
    cls_pos = np.sum(np.logical_and(-n_outer <= rise_dist, rise_dist < n_inner), axis=0) > 0
    cls_neg = np.sum(np.logical_and(-n_inner <= fall_dist, fall_dist < n_outer), axis=0) > 0
    cls_one = y.astype(bool)
    cls_num = np.select([cls_pos, cls_neg, cls_one], [1, 2, 3], default=0)

    import keras.utils.np_utils
    y = keras.utils.np_utils.to_categorical(cls_num)

    return y[SEQ_LEN - 1:]


if __name__ == '__main__':
    video_name = '20230205_04_Narumoto_Harimoto'

    fg = FeatureGenerator(video_name)
    f = fg.create2(with_label=True)
    g, h = f.feature, f.label
    print(f'{(g.shape, h.shape)=}')

    print(h.dtype, h.shape)

    TEST_SPLIT_POINT = 2900
    x_test, x_train = x_rnn_reshape(g[:TEST_SPLIT_POINT]), x_rnn_reshape(g[TEST_SPLIT_POINT:])
    y_test, y_train = y_rnn_reshape(h[:TEST_SPLIT_POINT]), y_rnn_reshape(h[TEST_SPLIT_POINT:])

    print(f'{(x_train.shape, y_train.shape)=}')

    n_hidden = 128

    if 0:
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense, Activation
        from tensorflow.keras.layers import LSTM
        from tensorflow.keras.optimizers import Adam
        from tensorflow.keras.callbacks import EarlyStopping

        model = Sequential()
        model.add(LSTM(
            n_hidden,
            batch_input_shape=(None, SEQ_LEN, g.shape[1]),
            return_sequences=False
        ))

        # model.add(Flatten())
        model.add(Dense(64))
        model.add(Activation("linear"))
        model.add(Dense(4))
        model.add(Activation("softmax"))
        optimizer = Adam(learning_rate=0.001)
        model.compile(loss="mean_squared_error", optimizer=optimizer)
        model.summary()

        early_stopping = EarlyStopping(monitor='val_loss', mode='auto', patience=40)
        model.fit(
            x_train, y_train,
            batch_size=2048,
            epochs=200,
            validation_split=0.1,
            callbacks=[early_stopping]
        )

        model.save('model')
    else:
        from tensorflow.keras.models import load_model

        model = load_model('model')
        model.summary()

    y_pred = model.predict(x_test)

    if 1:
        fig, axes = plt.subplots(4, 1, figsize=(100, 10), sharex=True)
        axes[0].imshow(np.tile(y_test[:, -1][:, None], 30).T)
        # axes[1].imshow(y_pred.T)
        # axes[1].set_aspect('auto')
        # axes[1].plot(y_pred[:, 0], color='black')
        axes[1].set_title('green red dup vaule')
        axes[1].plot(np.minimum(y_pred[:, 3], y_pred[:, 1]))
        axes[1].grid()
        axes[2].set_title('blue red dup vaule')
        axes[2].plot(np.minimum(y_pred[:, 2], y_pred[:, 1]))
        axes[2].grid()
        axes[3].plot(y_pred[:, 1], color='green', label='rise')  # rise
        axes[3].plot(y_pred[:, 2], color='blue', label='fall')  # fall
        axes[3].plot(y_pred[:, 3], color='red', label='rally')  # rally
        axes[3].legend()
        axes[-1].grid()

        from PIL import Image


        def generate_timestamp_ticklabel(ax):
            n = TEST_SPLIT_POINT - SEQ_LEN
            ax.set_xticks(
                np.linspace(0, n - 1, n // 10).round(0).astype(int)
            )
            lst = []
            for sec in f.timestamp[axes[-1].get_xticks() + SEQ_LEN]:
                lst.append(
                    f"{int(sec // 60):02}:{int(sec) % 60: 02}'{int((sec - int(sec)) * 100):02}")
            ax.set_xticklabels(lst, rotation=90)


        generate_timestamp_ticklabel(axes[-1])
        fig.tight_layout()
        fig.show()

    if 0:
        import async_writer
        from PIL import Image, ImageDraw
        import os
        from tqdm import tqdm

        with async_writer.AsyncVideoFrameWriter(os.path.expanduser('~/Desktop/out2.mp4'),
                                                30 / 3) as wr:
            with dataset.VideoBaseFrameStorage(
                    dataset.get_video_frame_dump_dir_path(video_name),
                    mode='r'
            ) as vf_store:
                vf_timestamp = vf_store.get_all_of('timestamp')
                vf_idx, = np.where(np.in1d(vf_timestamp, f.timestamp[SEQ_LEN - 1:TEST_SPLIT_POINT]))
                for j, i in tqdm(enumerate(vf_idx)):
                    data = vf_store.get(i)
                    img = Image.fromarray(data['original'])
                    draw = ImageDraw.Draw(img)
                    H = 10
                    draw.rectangle(xy=(H // 2, H // 2, 400 + H / 2, H * 3 + H // 2),
                                   fill=(255, 255, 255))
                    draw.rectangle(xy=(H // 2, H // 2, 200 + H / 2, H * 6 + H // 2),
                                   fill=(255, 255, 255))
                    B = int(255 * 0.6)
                    for k, s in enumerate(['RISE', 'POSITIVE', 'FALL']):
                        arg = [1, 3, 2][k]
                        R = 255 if y_pred[j, [1, 3, 2]].argmax() == k and y_pred[
                            j, arg] > 0.05 else 0
                        draw.text(xy=(H, H * k + H + H + H), fill=(R, 0, 0), text=s)
                    draw.text(xy=(H, H), fill=(0, 0, 0),
                              text=f'note_rnn_rally_detection.py GTX1060 3GB I={i:05} T={data["timestamp"]:6.1f}')
                    draw.text(xy=(H, H + H), fill=(0, 0, 0),
                              text='MODEL: SEQ50->[LSTM128->D64Lin->D4Sig]->4CLS[NEG8|2POS2|8NEG]')
                    draw.rectangle(
                        xy=(100, H * 0 + H * 3, 100 + y_pred[j, 1] * 100, H + H * 0 + H * 3),
                        fill=(B, B, 0))
                    draw.rectangle(
                        xy=(100, H * 1 + H * 3, 100 + y_pred[j, 3] * 100, H + H * 1 + H * 3),
                        fill=(0, B, 0))
                    draw.rectangle(
                        xy=(100, H * 2 + H * 3, 100 + y_pred[j, 2] * 100, H + H * 2 + H * 3),
                        fill=(0, 0, B))
                    # draw.rectangle(xy=(50, H * 3 + H, 50 + y_pred[j, 3] * 100, H + H * 3 + H),
                    #                fill=(B, 0, 0))
                    plt.figure()
                    wr.write(np.array(img))
