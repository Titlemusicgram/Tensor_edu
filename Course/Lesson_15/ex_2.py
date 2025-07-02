import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"

import matplotlib.pyplot as plt
import tensorflow as tf
from keras.utils import to_categorical
from keras.datasets import mnist
from keras.layers import Dense, Input
from keras.optimizers import Adam
from keras.metrics import CategoricalAccuracy
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Reshape, Conv2DTranspose, BatchNormalization
from keras.models import Model


tf.random.set_seed(1)

# Подготовка данных
# Загружаем набор данных cifar10 и разбиваем его по переменным
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Нормируем значения, чтобы были от 0 до 1
x_train = x_train / 255.0
x_test = x_test / 255.0

# Преобразуем y_train в вектор. Пример формата - [0, 0, 1, 0, 0, 0, 0, 0, 0, 0] (это для значения y_train = 3)  
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)


# Опишем модель
enc_input = Input(shape=(28, 28, 1))
x = Conv2D(32, 3, activation='relu')(enc_input)
x = MaxPooling2D(2, padding='same')(x)
x = Conv2D(64, 3, activation='relu')(x)
x = MaxPooling2D(2, padding='same')(x)
x = Flatten()(x)
hiden_output = Dense(8, activation='linear')(x)

x = Dense(7 * 7 * 8, activation='relu')(hiden_output)
x = Reshape((7, 7, 8))(x)
x = Conv2DTranspose(64, 5, strides=(2, 2), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = Conv2DTranspose(32, 5, strides=(2, 2), activation='linear', padding='same')(x)
x = BatchNormalization()(x)
dec_output = Conv2DTranspose(1, 3, activation='sigmoid', padding='same', name='dec_output')(x)

x2 = Dense(128, activation='relu')(hiden_output)
class_output = Dense(10, activation='softmax', name='class_output')(x2)


model = Model(enc_input, [dec_output, class_output])


model.compile(
    optimizer=Adam(learning_rate=0.01),
    # loss=['mean_squared_error', 'categorical_crossentropy'], # зададим разные функции потерь для каждого выхода
    # metrics=['accuracy', 'accuracy'] # тоже самое с метриками 
    loss={
        'dec_output': 'mean_squared_error',
        'class_output': 'categorical_crossentropy'
    }, # также их можно задать явно для каждого выхода с помощью словаря
    loss_weights={
        'dec_output': 1.0,
        'class_output': 0.5
    }, # здесь же можно задавать веса для потерь как списком, так и словарем
    metrics= {
        'dec_output': None,
        'class_output': 'accuracy'
    } # аналогично можно задавать метрики
)


model.fit(x_train, {'dec_output': x_train, 'class_output': y_train}, batch_size=64, epochs=1) # в методк fit тоже можно задавать выходы словарем

p = model.predict(tf.expand_dims(x_test[0], axis=0))

print(tf.argmax(p[1], axis=1).numpy())

plt.subplot(121)
plt.imshow(x_test[0], cmap='gray')
plt.subplot(122)
plt.imshow(p[0].squeeze(), cmap='gray')
plt.show()
