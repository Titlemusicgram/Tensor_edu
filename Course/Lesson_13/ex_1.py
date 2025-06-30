import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"

import tensorflow as tf
from keras.utils import to_categorical
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, MaxPooling2D, Dense, Dropout, add
from keras.models import Model
from keras.datasets import cifar10

tf.random.set_seed(1)

# ResNet-подобная сеть

# Подготовка данных
# Загружаем набор данных cifar10 и разбиваем его по переменным
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Нормируем значения, чтобы были от 0 до 1
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Преобразуем y_train в вектор. Пример формата - [0, 0, 1, 0, 0, 0, 0, 0, 0, 0] (это для значения y_train = 3)  
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)


# Создадим сеть
# Опишем слои сети
input_layer = Input(shape=(32, 32, 3), name='img')
x = Conv2D(32, 3, activation='relu')(input_layer)
x = Conv2D(64, 3, activation='relu')(x)
block_1_output = MaxPooling2D(3)(x)

x = Conv2D(64, 3, activation='relu', padding='same')(block_1_output)
x = Conv2D(64, 3, activation='relu', padding='same')(x)
block_2_output = add([x, block_1_output])

x = Conv2D(64, 3, activation='relu', padding='same')(block_2_output)
x = Conv2D(64, 3, activation='relu', padding='same')(x)
block_3_output = add([x, block_2_output])

x = Conv2D(64, 3, activation='relu')(block_3_output)
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
output_layer = Dense(10, activation='softmax')(x)

model = Model(input_layer, output_layer, name='toy_resnet')

# Соберем модель
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Обучим модель
model.fit(x_train, y_train, batch_size=64, epochs=15, validation_split=0.2)

# Проверим модель на тестовых данных
a = model.evaluate(x_test, y_test)
print(a)
