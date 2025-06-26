import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"

import tensorflow as tf

# Математические функции
print('Матиматические функции\n')

a = tf.constant([1, 2, 3])
b = tf.constant([9, 8, 7])

c = tf.add(a, b) #размерности тензоров должны быть одинаковыми
print(f'{c}\n')

d = a + b #тоже складывает. нет разницы
print(f'{d}\n')

e = tf.subtract(a, b)
print(f'{e}\n')

f = a - b
print(f'{f}\n')

g = tf.divide(a, b)
print(f'{g}\n')

h = a / b
print(f'{h}\n')

i = tf.multiply(a, b)
print(f'{i}\n')

j = a * b
print(f'{j}\n')

k = a ** 2
print(f'{k}\n')

l = tf.tensordot(a, b, axes=0) #внешнее векторное умножение, получаем матрицу
print(f'{l}\n')

m = tf.tensordot(a, b, axes=1) #внутреннее векторное умножение, получаем число
print(f'{m}\n')

n = tf.constant(tf.range(1, 10), shape=(3, 3))
o = tf.constant(tf.range(5, 14), shape=(3, 3))
p = tf.matmul(n, o) #матричное умножение
print(f'{p}\n')

q = n @ o #тоже матричное умножение
print(f'{q}\n')

r = tf.reduce_sum(q) #сумма всех значений матрицы
print(f'{r}\n')

s = tf.reduce_sum(q, axis=0) #сумма значений по столбцам
print(f'{s}\n')

t = tf.reduce_mean(q) #среднее арифметическое из всех значений матрицы
print(f'{t}\n')

u = tf.reduce_max(q) #максимальное значение в матрице, минимальное аналогично. Подобные функции все работают с задаванием осей axes
print(f'{u}\n')

v = tf.reduce_prod(q, axis=1) #вычисляет произведение всех элементов по строкам.
print(f'{v}\n')

w = tf.sqrt(tf.cast(q, dtype=tf.float32)) #корень из всех элементов матрицы. Сначала приводим к типу float, т.к. работает только с ним
print(f'{w}\n')

x = tf.square(q) #квадрат
print(f'{x}\n')

y = tf.sin(w) #работает вся тригонометрия. Тоже сначала привести к float
print(f'{y}\n')

from tensorflow.python.keras.losses import categorical_crossentropy
from tensorflow.python.keras import optimizers
