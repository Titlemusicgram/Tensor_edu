import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"

import tensorflow as tf
import keras
from keras.layers import Dense, InputLayer
from keras.models import Sequential, Model
from keras.utils import to_categorical
from keras.datasets import mnist


# Подготовка данных
# Загружаем набор данных mnist и разбиваем его по переменным
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Нормируем значения, чтобы были от 0 до 1
x_train = x_train /255
x_test = x_test /255

# Вытягиваем все изображения 28 на 28 пикселей в единый вектор
x_train = tf.reshape(tf.cast(x_train, tf.float32), [-1, 28*28])
x_test = tf.reshape(tf.cast(x_test, tf.float32), [-1, 28*28])

# Преобразуем y_train в вектор. Пример формата - [0, 0, 1, 0, 0, 0, 0, 0, 0, 0] (это для значения y_train = 3)  
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)


# Создаем модель
model = Sequential([
    InputLayer(shape=(28*28,)),
    Dense(128, activation='relu', name='layer_1'),
    Dense(10, activation='softmax', name='layer_2'),
])


# Собираем нейросеть
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)


# Тренируем нейросеть
model.fit(x=x_train, y=y_train, batch_size=32, epochs=5)

# Проверяем на тестовой выборке
test_results = model.evaluate(x=x_test, y=y_test, batch_size=32)
print(f'\n{test_results}')


# Создадим на основе нашей модели модель, у которой входы будут такими же, а выходы после каждого слоя
model_ex = Model(
    inputs=model.inputs,
    outputs=[layer.output for layer in model.layers]
)


# Проверим нашу новую модель. Она сразу уже обучена, т.к. ссылается на первую
x = tf.expand_dims(x_test[0], axis=0)
y = model_ex(x)
y2 = model(x)
print(y, y2, sep='\n\n')


# Можем добавлять слои к уже готовой модели
model_ex_2 = Sequential([
    model,
    Dense(10, activation='tanh')
])


# Обучим эту вторую модель, но так, чтобы первая не обучалась еще раз
model.trainable = False
model_ex_2.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
model_ex_2.fit(x=x_train, y=y_train, epochs=3, batch_size=32)
