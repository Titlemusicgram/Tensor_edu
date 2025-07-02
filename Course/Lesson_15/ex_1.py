import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"

import tensorflow as tf
import keras
from keras.utils import to_categorical
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Input
from keras.optimizers import Adam
from keras.metrics import CategoricalAccuracy
from keras.losses import CategoricalCrossentropy


tf.random.set_seed(1)

# Подготовка данных
# Загружаем набор данных mnist и разбиваем его по переменным
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Нормируем значения, чтобы были от 0 до 1
x_train = x_train.reshape(-1, 28 * 28) / 255.0
x_test = x_test.reshape(-1, 28 * 28) / 255.0

# Преобразуем y_train в вектор. Пример формата - [0, 0, 1, 0, 0, 0, 0, 0, 0, 0] (это для значения y_train = 3)  
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Описываем модель
model = Sequential([
    Input(shape=(28 * 28,)),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

# Опишем свою функцию потерь через функцию
# Обязательно используем здесь только функции TensorFlow
def my_loss(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true-y_pred))


# Опишем свою функцию потерь через класс
# Обязательно используем здесь только функции TensorFlow
class My_loss(keras.losses.Loss):
    def __init__(self, alpha=1.0, beta=1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def call(self, y_true, y_pred):
        return tf.reduce_mean(tf.square(self.alpha * y_true- self.beta * y_pred))


# Опишем свою метрику через класс
class CategoricalTruePositives(keras.metrics.Metric):
    def __init__(self, name='my_metric'):
        super().__init__(name=name)
        self.true_positives = self.add_weight(name='acc', initializer='zeros')
        self.count = tf.Variable(0.0)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.reshape(tf.argmax(y_pred, axis=1), shape=(-1, 1))
        y_true = tf.reshape(tf.argmax(y_true, axis=1), shape=(-1, 1))
        values = tf.cast(y_true, 'int32') == tf.cast(y_pred, 'int32')

        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, 'float32')
            values = tf.multiply(values, sample_weight)

        values = tf.cast(values, 'float32')

        self.true_positives.assign_add(tf.reduce_mean(values))
        self.count.assign_add(1.0)

    def result(self):
        return self.true_positives/self.count
    
    def reset_state(self):
        self.true_positives.assign(0.0)
        self.count.assign(0.0)


model.compile(
    optimizer=Adam(learning_rate=0.001),
    # loss=CategoricalCrossentropy(),
    # loss=my_loss, # зададим потери через описанную выше функцию потерь
    loss=My_loss(alpha=0.5, beta=2.0), # зададим потери через описанный выше класс потерь
    metrics=[
        CategoricalAccuracy(),
        CategoricalTruePositives() # добавим метрику описанную выше. Результаты должны быть одинаковыми
    ]
)

history = model.fit(x_train, y_train, batch_size=64, epochs=3, validation_split=0.2)
print(model.evaluate(x_test, y_test), end='\n\n')
