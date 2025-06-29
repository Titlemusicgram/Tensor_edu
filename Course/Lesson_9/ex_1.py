import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"

import tensorflow as tf
from keras.layers import Layer


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
class NeuralNetwork(Layer):
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


# Создаем экземпляр слоя и проверяем считает ли он

# layer_1 = DenseLayer(10)
# y = layer_1(tf.constant([[1., 2., 3.]]))
# print(y)


# Создаем экземпляр модели
model = NeuralNetwork()
y = model(tf.constant([[1.0, 2.0, 3.0]]))
print(y)
