import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"

import tensorflow as tf

# Автозаполнение
print('Автозаполнение\n')
a = tf.zeros((3, 3))
print(f'{a.numpy()}\n')

b = tf.ones((3, 3))
print(f'{b.numpy()}\n')

с = tf.ones_like(a)
print(f'{с.numpy()}\n')

d = tf.zeros_like(b)
print(f'{d.numpy()}\n')

e = tf.eye(3)
print(f'{e.numpy()}\n')

f = tf.eye(3, 2)
print(f'{f.numpy()}\n')

g = tf.identity(a)
print(f'{g.numpy()}\n')

h = tf.fill((4, 2), -1)
print(f'{h.numpy()}\n')

i = tf.range(1, 10, 0.5)
print(f'{i.numpy()}\n')

# Генерация тензоров со случайными значениями
print('Генерация тензоров со случайными значениями\n')
j = tf.random.normal((3, 3), mean=0, stddev=2)
print(f'{j.numpy()}\n')

k = tf.random.truncated_normal((3, 3), mean=0, stddev=2)
print(f'{k.numpy()}\n')

tf.random.set_seed(23)
l = tf.random.uniform((4, 5), 0, 10, dtype=tf.int32)
print(f'{l.numpy()}\n')

tf.random.set_seed(23)
m = tf.random.uniform((4, 5), 0, 10, dtype=tf.int32)
print(f'{m.numpy()}\n') #из-за функции set_seed тензоры получились одинаковыми. Ее необходимо вызывать каждый раз
