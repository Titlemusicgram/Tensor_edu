import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"

import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, InputLayer


# Вариант № 1 задавать структуру нейронной сети:
model = Sequential(
    layers=[
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ]
)

# Показывает слои модели
print(model.layers)

# Можно удалять слои можели
model.pop()
print(model.layers)

# Можно добавлять слои
model.add(Dense(10, activation='linear', name='layer_2'))
print(f'{model.layers}\n')

# Вариант № 2 задавать структуру нейронной сети:
model_2 = Sequential()
model_2.add(Dense(128, activation='relu', name='layer_1'))
model_2.add(Dense(10, activation='softmax', name='layer_2'))

print(model_2.layers)

# Можно обращаться к конкретным слоям модели
weights = model_2.layers[0].weights
print(weights) # пока пустой список, потому-что они инициализируются после первого пропускания данных через сеть

x = tf.random.uniform((1, 20), 0, 1)
y = model_2(x)
weights = model_2.layers[1].weights
print(f'{weights}\n') # здесь они уже есть

# Выводим в консоль сводку по сети
model_2.summary()
print('\n')

# Можно строить нейросеть с входным слоем Input. Тогда веса сразу будут
model_3 = Sequential([
    InputLayer((20, )),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

model_3.summary() # количество слоев все еще 2. Input не отображается
