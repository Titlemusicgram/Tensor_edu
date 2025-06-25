import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"

import tensorflow as tf


print('\n№1 Тензор размером в 2 строки по 3 столбца, заполненный нулями типа int\n\nКак объект:')
zero_tensor = tf.zeros([2, 3], dtype=tf.int32)
print(zero_tensor) #выводит объект тензора

print('\nКак значение:')
print(zero_tensor.numpy()) #выводит список тензора, то есть его значение

print('\n№2 Тензор размером в 1 строку из 2 столбцов, заполненный единицами типа float')
ones_tensor = tf.ones(2)
print(ones_tensor.numpy())

print('\n№3 Тензор размером в 2 строчки по 4 столбца с одинаковыми значениями, которые задаем руками. ' \
'Тип данных берется из заданного значения')
fives_tensor = tf.fill([2,4], value=5) 
print(fives_tensor.numpy())

print('\n№4 Тензор размером в 2 строку из 3 столбцов, заполненный данными вручную. ' \
'Не меняется, так как константа')
const_tensor = tf.constant([[1, 2, 3], [4, 5, 6]])
print(const_tensor.numpy())

print('\n№5 Тензор, заполненный случайными числами со средним 10 и стандартным отклонением в 5')
random_tensor = tf.random.normal([3, 3], mean=10, stddev=5)
print(random_tensor.numpy())

print('\n№6 Тензор, заполненный случайными числами со средним 0 и стандартным отклонением в 100, ' \
'и обрезанный по 2-м стандартным отклонениям от среднего. Считается общепринятым')
truncated_tensor = tf.random.truncated_normal([2, 5], mean=0, stddev=100)
print(truncated_tensor.numpy())

print('\n№7 Тензор, заполненный случайными числами выбранными из заданного диапазона. ' \
'Распределение равномерное')
uniform_tensor = tf.random.uniform([3,2], minval=0, maxval=10, dtype=tf.int32)
print(uniform_tensor.numpy())
