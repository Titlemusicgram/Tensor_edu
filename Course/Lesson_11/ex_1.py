import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"

import tensorflow as tf
import keras
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from keras.models import Model
from keras import Input
from keras.datasets import cifar10
from keras.utils import to_categorical


# Функциональный подход к построению модели

# Уберем рандом, чтобы результаты повторялись
tf.random.set_seed(1)


# Подготовка данных
# Загружаем набор данных cifar10 и разбиваем его по переменным
(x_train, y_train), (x_test, y_test) = cifar10.load_data()


# Нормируем значения, чтобы были от 0 до 1
x_train = x_train /255
x_test = x_test /255

# Преобразуем y_train в вектор. Пример формата - [0, 0, 1, 0, 0, 0, 0, 0, 0, 0] (это для значения y_train = 3)  
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)


# Модель
# Опишем связи слоев функциональным способом
input = Input(shape=(32, 32, 3))
x = Conv2D(32, 3, activation='relu')(input)
x = MaxPooling2D(2, padding='same')(x)
x = Conv2D(64, 3, activation='relu')(x)
x = MaxPooling2D(2, padding='same')(x)
x = Flatten()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(10, activation='softmax')(x)

# Создадим модель
model = Model(input, output)

# model.summary()

# Собираем модель
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Проводим обучение
model.fit(x=x_train, y=y_train, batch_size=64, epochs=10, validation_split=0.2)

# Проверим на тестовых данных
print(model.evaluate(x=x_test, y=y_test))
