import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"

import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, Reshape, Conv2DTranspose, BatchNormalization
from keras.models import Model
from keras import Input
from keras.datasets import mnist
from keras.utils import to_categorical


# Модель енкодера-декодера функциональным способом

# Подготовка данных
# Загружаем набор данных cifar10 и разбиваем его по переменным
(x_train, y_train), (x_test, y_test) = mnist.load_data()


# Нормируем значения, чтобы были от 0 до 1
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Преобразуем y_train в вектор. Пример формата - [0, 0, 1, 0, 0, 0, 0, 0, 0, 0] (это для значения y_train = 3)  
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)


# Опишем модель енкодера
enc_input = Input(shape=(28, 28, 1))
x = Conv2D(32, 3, activation='relu')(enc_input)
x = MaxPooling2D(2, padding='same')(x)
x = Conv2D(64, 3, activation='relu')(x)
x = MaxPooling2D(2, padding='same')(x)
x = Flatten()(x)
enc_output = Dense(8, activation='linear')(x)

encoder = Model(enc_input, enc_output, name='encoder')


# Опишем модель декодера
dec_input = Input(shape=(8,), name='encoded_img')
x = Dense(7 * 7 * 8, activation='relu')(dec_input)
x = Reshape((7, 7, 8))(x)
x = Conv2DTranspose(64, 5, strides=(2, 2), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = Conv2DTranspose(32, 5, strides=(2, 2), activation='linear', padding='same')(x)
x = BatchNormalization()(x)
dec_output = Conv2DTranspose(1, 3, activation='sigmoid', padding='same')(x)

decoder = Model(dec_input, dec_output, name='decoder')


# Свяжем енкодер с декодером
autoencoder_input = Input(shape=(28, 28, 1), name='img')
x = encoder(autoencoder_input)
autoencoder_output = decoder(x)

autoencoder = Model(autoencoder_input, autoencoder_output, name='autoencoder')


# Соберем модель
autoencoder.compile(
    optimizer='adam',
    loss='mean_squared_error'
)


# Запустим тренировку
autoencoder.fit(x_train, x_train, batch_size=32, epochs=1)

# Проверим как работает на одном изображении
h = encoder.predict(tf.expand_dims(x_test[0], axis=0))
img = decoder.predict(h)

plt.subplot(121)
plt.imshow(x_test[0], cmap='gray')
plt.subplot(122)
plt.imshow(img.squeeze(), cmap='gray')
plt.show()
