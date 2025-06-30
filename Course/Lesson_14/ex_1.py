import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"

import numpy as np
import tensorflow as tf
from keras.utils import to_categorical
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Input
from tensorflow.python.data import Dataset 
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model


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

# (ОПЦИЯ)Опишем валидационную выборку руками
validation_split = 0.2
validation_split_index = np.ceil(x_train.shape[0] * validation_split).astype('int32')

x_train_val = x_train[:validation_split_index]
y_train_val = y_train[:validation_split_index]

x_train_data = x_train[validation_split_index:]
y_train_data = y_train[validation_split_index:]

# (ОПЦИЯ)Опишем тренировочный и валидационный сеты данных иным способом
train_dataset = Dataset.from_tensor_slices((x_train_data, y_train_data))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(64) # перемешали и разбили по батчам

val_dataset = Dataset.from_tensor_slices((x_train_val, y_train_val))
val_dataset = val_dataset.batch(64)

# (ОПЦИЯ) Количество данных по каждой категории должно быть одинаковым дл нормального обучения. Если такого добиться нельзя, то модно задать веса как показано здесь.
# Каких данных меньше, те веса и увеличиваем
class_weight = {
    0: 1.0,
    1: 1.0,
    2: 1.0,
    3: 1.0,
    4: 1.0,
    5: 1.0,
    6: 1.0,
    7: 1.0,
    8: 1.0,
    9: 1.0
}

# (ОПЦИЯ) Зададим всем изображениям вес 1.0, а 5-му вес 2.0 
sample_weight = np.ones(shape=len(x_train)) # задали всем изображениям вес 1
sample_weight[5] = 2.0

# Описываем модель
model = Sequential([
    Input(shape=(28 * 28,)),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Callback EarlyStopping позволяет останавливать обучение, если оно идет не очень хорошо, чтобы не тратить время
# Так же часто используется Callback ModelCheckpoint для сохранения модели с заданной регулярностью
callbacks = [
    EarlyStopping(
        monitor='loss',
        min_delta=0.01,
        patience=2,
        verbose=1
    ),
    ModelCheckpoint(
        filepath='model_{epoch}.keras',
        save_best_only=True,
        monitor='loss',
        verbose=1
    )
]


# Обучение с включенным коллбеком
history = model.fit(x_train, y_train, batch_size=64, epochs=3, validation_split=0.2, callbacks=callbacks)
print(history.history) # переменная history хванит в себе стаститику по обучению, которую можно посмотреть как показано здесь

# Если мы руками задали веса целым категориям как было выше
# model.fit(x_train, y_train, batch_size=64, epochs=5, validation_split=0.2, class_weight=class_weight)

# Если мы руками задали веса отдельным фотографиям (выше пример)
# model.fit(x_train, y_train, batch_size=64, epochs=5, validation_split=0.2, sample_weight=sample_weight)

# Если валидационная выборка описана руками
# model.fit(x_train_data, y_train_data, batch_size=64, epochs=5, validation_data=(x_train_val, y_train_val))

# Если мы описали тренировочный и валидационный сеты руками
# model.fit(train_dataset, epochs=5, validation_data=val_dataset)

# Если мы указываем явно шаги в одной эпохе (нужно следить, чтобы данных хватило)
# model.fit(train_dataset, epochs=5, steps_per_epoch=100, validation_data=val_dataset)

# Если мы указываем явно шаги в одной эпохе в проверочной выборке
# model.fit(train_dataset, epochs=5, validation_steps=5, validation_data=val_dataset)

print(model.evaluate(x_test, y_test), end='\n\n')


# Поскольку мы сохраняли модели, их можно загрузить
model_new = load_model(filepath='model_3.keras')
print(model_new.evaluate(x_test, y_test))
