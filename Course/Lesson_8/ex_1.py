import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras import optimizers
from keras.utils import to_categorical
from keras.ops.nn import relu, softmax
from keras.datasets import mnist
from keras.losses import categorical_crossentropy
from keras.metrics import Accuracy


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


# Создание нейронной сети

# Создание модель нейронной сети
class DenseNN(tf.Module):
    def __init__(self, outputs, activate='relu'):
        super().__init__()
        self.outputs = outputs
        self.activate = activate
        self.fl_init = False

    def __call__(self, x):
        if not self.fl_init:
            self.w = tf.random.truncated_normal((x.shape[-1], self.outputs), stddev=0.1, name='w')
            self.b = tf.zeros([self.outputs], dtype=tf.float32, name='b')

            self.w = tf.Variable(self.w)
            self.b = tf.Variable(self.b)

            self.fl_init = True

        y = x @ self.w + self.b

        if self.activate == 'relu':
            return relu(y)
        elif self.activate == 'softmax':
            return softmax(y)
        
        return y


# Создаем класс, который будет отображать всю нейронную сеть
class SequentialModule(tf.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = DenseNN(128)
        self.layer_2 = DenseNN(10, activate='softmax')

    def __call__(self, x):
        return self.layer_2(self.layer_1(x))
    

model = SequentialModule()


# Обучаем нейронную сеть
cross_entropy = lambda y_true, y_pred: tf.reduce_mean(categorical_crossentropy(y_true, y_pred))
opt = optimizers.Adam(learning_rate=0.001)

BATCH_SIZE = 32
EPOCHS = 10
TOTAL = x_train.shape[0]

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(BATCH_SIZE)

# Перенесем вычисления на уровень графа (ОБРАТИТЬ ВНИМАНИЕ НАСКОЛЬКО БЫСТРЕЕ ТРЕНИРУЕТСЯ НЕФРОСЕТЬ ПО СРАВНЕНИЮ С УРОКОМ 7!!!!!!!!!!!!!!)
@tf.function
def train_batch(x_batch, y_batch):
    with tf.GradientTape() as tape:
        f_loss = cross_entropy(y_batch, model(x_batch))

    grads = tape.gradient(f_loss, model.trainable_variables)
    opt.apply_gradients(zip(grads, model.trainable_variables))

    return f_loss


for n in range(EPOCHS):
    loss = 0

    for x_batch, y_batch in train_dataset:
        loss += train_batch(x_batch, y_batch)

    print(loss.numpy())


# Определяем качество работы нейросети на тестовой выборке
y = model(x_test)
y2 = tf.argmax(y, axis=1).numpy()
acc = len(y_test[y_test == y2]) / y_test.shape[0] * 100
print(acc)
