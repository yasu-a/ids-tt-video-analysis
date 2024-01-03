# https://qiita.com/mako0715/items/b6605a77467ac439955b

import tensorflow as tf

# 0~9の手書き文字MNISTのデータセットを読み込む
(training_images, training_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
# 画像データの形式を変更する
training_images = training_images.reshape(training_images.shape[0], 28, 28, 1)
test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)
# 画像データを正規化する
training_images = training_images / 255.0
test_images = test_images / 255.0
# ラベルデータを1-of-K表現にする
training_labels = tf.keras.utils.to_categorical(training_labels)
test_labels = tf.keras.utils.to_categorical(test_labels)
# CNNのモデルを作成する
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])
# 任意のオプティマイザと損失関数を設定してモデルをコンパイルする
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# ネットワーク各層の出力内容を確認する
model.summary()
# モデルをトレーニングする
model.fit(training_images, training_labels, epochs=5)
# テストデータで精度を確認する
test_loss = model.evaluate(test_images, test_labels)
