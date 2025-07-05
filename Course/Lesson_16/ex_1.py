import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"

import tensorflow as tf
import keras
from keras.layers import Dense
from keras.models import Sequential
from keras.utils import to_categorical
from keras.datasets import mnist
from keras.models import load_model

# Подготовка данных
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(-1, 784) / 255.0
x_test = x_test.reshape(-1, 784) / 255.0

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)


# Создаем и тренируем модель
model = Sequential([
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5)

# Сохраним и загрузим модель
model.save('model_saved.keras')

model_loaded = load_model('model_saved.keras')

model_loaded.evaluate(x_test, y_test)
