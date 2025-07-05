import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"

import tensorflow as tf
import keras
from keras.layers import Dense, TFSMLayer
from keras.models import Sequential
from keras.utils import to_categorical
from keras.datasets import mnist
from keras.models import load_model, Model


# Подготовка данных
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(-1, 784) / 255.0
x_test = x_test.reshape(-1, 784) / 255.0

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)


# Кастомизируем модель
@keras.saving.register_keras_serializable()
class NeuralNetwork(Model):
    def __init__(self, units):
        super().__init__()
        self.units = units
        self.model_layers = [Dense(n, activation='relu') for n in self.units]

    def call(self, inputs):
        x = inputs
        for layer in self.model_layers:
            x = layer(x)
        return x
    
    def get_config(self):
        return {"units": self.units}
        

# Создадим свою модель. Поправим в ней функцию активации на линейную
@keras.saving.register_keras_serializable()
class NeuralNetworkLinear(Model):
    def __init__(self, units):
        super().__init__()
        self.units = units
        self.model_layers = [Dense(n, activation='linear') for n in self.units]

    def call(self, inputs):
        x = inputs
        for layer in self.model_layers:
            x = layer(x)
        return x
    
    def get_config(self):
        return {"units": self.units}


model = NeuralNetwork([128, 10])

y = model.predict(tf.expand_dims(x_test[0], axis=0))
print(y)


# Сохраним и загрузим кастомную модель
model.save('saved_model_3.keras')
# При загрузке заменим класс на наш собственный
model_loaded = load_model('saved_model_3.keras', custom_objects={'NeuralNetwork': NeuralNetworkLinear})


y2 = model_loaded.predict(tf.expand_dims(x_test[0], axis=0))
print(y2)

# Методы весов
weights = model.get_weights()
model_loaded.set_weights(weights)
keras.saving.save_weights(model=model,filepath='weights.weights.h5')
keras.saving.load_weights(model=model_loaded, filepath='weights.weights.h5')
