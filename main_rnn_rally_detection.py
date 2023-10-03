import matplotlib.pyplot as plt

import numpy as np

np.set_printoptions(suppress=True)

import numpy as np

np.set_printoptions(suppress=True)

from feature_generator import FeatureGenerator

import numpy as np

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
    y = y[SEQ_LEN - 1:][:, None]
    return np.concatenate([~y, y], axis=1)


if __name__ == '__main__':
    fg = FeatureGenerator('20230205_04_Narumoto_Harimoto')
    f = fg.create(label=True)
    g, h = f['feature'], f['label']
    print(f'{(g.shape, h.shape)=}')

    print(h.dtype, h.shape)

    TEST_SPLIT_POINT = 9000
    x_train, x_test = x_rnn_reshape(g[:TEST_SPLIT_POINT]), x_rnn_reshape(g[TEST_SPLIT_POINT:])
    y_train, y_test = y_rnn_reshape(h[:TEST_SPLIT_POINT]), y_rnn_reshape(h[TEST_SPLIT_POINT:])

    print(f'{(x_train.shape, y_train.shape)=}')

    length_of_sequence = SEQ_LEN
    n_feature = g.shape[1]
    n_hidden = 128

    if 1:
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense, Activation
        from tensorflow.keras.layers import LSTM
        from tensorflow.keras.optimizers import Adam
        from tensorflow.keras.callbacks import EarlyStopping

        model = Sequential()
        model.add(LSTM(
            n_hidden,
            batch_input_shape=(None, length_of_sequence, n_feature),
            return_sequences=False
        ))

        # model.add(Flatten())
        model.add(Dense(64))
        model.add(Activation("linear"))
        model.add(Dense(2))
        model.add(Activation("softmax"))
        model.add(Dense(2))
        model.add(Activation("softmax"))
        optimizer = Adam(learning_rate=0.001)
        model.compile(loss="mean_squared_error", optimizer=optimizer)
        model.summary()

        early_stopping = EarlyStopping(monitor='val_loss', mode='auto', patience=20)
        model.fit(
            x_train, y_train,
            batch_size=1024,
            epochs=100,
            validation_split=0.1,
            callbacks=[early_stopping]
        )

        model.save('model')
    else:
        from tensorflow.keras.models import load_model

        model = load_model('model')
        model.summary()

    y_pred = model.predict(x_test)
    fig, axes = plt.subplots(2, 1, figsize=(100, 5), sharex=True)
    axes[0].imshow(np.tile(y_test[:, 1][:, None], 30).T)
    # axes[1].imshow(y_pred.T)
    # axes[1].set_aspect('auto')
    # axes[1].plot(y_pred[:, 0], color='black')
    axes[1].plot(y_pred[:, 1], color='red')
    fig.tight_layout()
    fig.show()
