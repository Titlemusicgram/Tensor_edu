import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"

import tensorflow as tf
import keras
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from keras.models import Model
from keras import Input
from keras.datasets import cifar10
from keras.utils import to_categorical


# Функциональный подход к построению модели через Tensorflow

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
# Опишем слой
class TfConv2D(tf.Module):
    def __init__(self, kernel=(3, 3), channels=1, strides=(2, 2), padding='SAME', activate='relu'):
        super().__init__()
        self.kernel = kernel
        self.channels= channels
        self.strides = strides
        self.padding = padding
        self.activate = activate
        self.fi_init = False

    def __call__(self, x):
        if not self.fi_init:
            self.w = tf.random.truncated_normal((*self.kernel, x.shape[-1], self.channels), stddev=0.1, dtype=tf.double)
            self.b = tf.zeros([self.channels], dtype=tf.double)

            self.w = tf.Variable(self.w)
            self.b = tf.Variable(self.b)

            self.fi_init = True

        y = tf.nn.conv2d(x, self.w, strides=(1, *self.strides, 1), padding=self.padding) + self.b

        if self.activate == 'relu':
            return tf.nn.relu(y)
        elif self.activate == 'softmax':
            return tf.nn.softmax(y)
        
        return y


# Создаем слои
layer_1 = TfConv2D((3, 3), 32)
y = layer_1(tf.expand_dims(x_test[0], axis=0))
print(y.shape, end='\n\n')

y = tf.nn.max_pool2d(y, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding="SAME")
print(y.shape)
