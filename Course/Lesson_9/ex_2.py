import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"

import tensorflow as tf
import keras
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.layers import Layer
from keras.models import Model


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


# Описываем слой
class DenseLayer(Layer):
    def __init__(self, units=1, activity_regularizer=None, trainable=True, dtype=None, autocast=True, name=None, **kwargs):
        super().__init__(activity_regularizer=activity_regularizer, trainable=trainable, dtype=dtype, autocast=autocast, name=name, **kwargs)
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer='random_normal',
            trainable=True
        )
        
        self.b = self.add_weight(shape=(self.units,), initializer='zeros', trainable=True)

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b
    

# Создаем нейросеть через наследование от слоя
class NeuralNetwork(Model):
    def __init__(self, *, activity_regularizer=None, trainable=True, dtype=None, autocast=True, name=None, **kwargs):
        super().__init__(activity_regularizer=activity_regularizer, trainable=trainable, dtype=dtype, autocast=autocast, name=name, **kwargs)
        self.layer_1 = DenseLayer(128)
        self.layer_2 = DenseLayer(10)

    def call(self, inputs):
        x = self.layer_1(inputs)
        x = tf.nn.relu(x)
        x = self.layer_2(x)
        x = tf.nn.softmax(x)
        return x


# Создаем экземпляр модели
model = NeuralNetwork()

# Собираем модель
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss=keras.losses.categorical_crossentropy,
    metrics=['accuracy', 'mean_squared_error']
)

# Тренируем модель
model.fit(x_train, y_train, batch_size=32, epochs=5)

# Проверяем на тестовой выборке
print(model.evaluate(x_test, y_test, batch_size=32))
