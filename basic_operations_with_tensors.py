import tensorflow as tf


tf.ones(1) # просто чтобы информационное сообщение не лезло в вывод

print('\n№1 Сложение тензоров можно делать обычным оператором. Сложим 2 единичных тензора и получим двойки')
a = tf.ones([2,2], dtype=tf.int32)
b = tf.ones([2,2], dtype=tf.int32)
c = a + b
print(c.numpy())

print('\n№2 Аналогично с умножением. Умножим единичный вектор на 3')
d = a * 3
print(d.numpy())

print('\n№3 При перемножении тензоров между собой выполняется не матричное умножение, а поэлементное. ' \
'Умножим матрицу с 2-ками на матрицу с 7-ками')
e = tf.fill([2,2], value=2)
f = tf.fill([2,2], value=7)
g = e * f
print(g.numpy())

print('\nМатричные операции\n')
print('\n№4 Создание квадратной единичной матрицы(матрица с 0 везде, кроме главной диагонали. Там единицы.) с помощью функции eye ' \
'Агрумент функции - число элементов в строчках и столбцах, так как она квадратная. Если задавать по отдельности количество строк и колонок,' \
'то все равно создастся квадратная матрица размерностью с наименьшим из заданных чисел)')
h = tf.eye(3, dtype=tf.int32)
print(h.numpy())

print('\n№5 Можно создавать векторы с помощью функции range. Создадим вектор, значения которого от 1 до 4')
i = tf.range(start=1, limit=5, delta=1)
print(i.numpy())

print('\n№6 А теперь созданный вектор можно передать в функцию создания диагональных матриц (tf.linalg.diag), чтобы определить значения по главной диагонали тензора')
j = tf.linalg.diag(i)
print(j.numpy())

print('\n№7 Существует так же функция вытаскивания диагонали из матрицы tf.linalg.diag_part')
k = tf.linalg.diag_part(j)
print(k.numpy())

print('\n№8 Транспонирование матрицы в 2 строки из 4-х столбцов функцией transpose')
l = tf.transpose(tf.ones([2, 4], dtype=tf.int32))
print(l.numpy())

print('\n№9 Матричное перемножение функцией matmul. Перемножим 2 единичные матрицы размерами 2 по 3 и 3 по 4')
m = tf.ones([2, 3])
print(f'{m.numpy()}\n')
n = tf.ones([3, 4])
print(f'{n.numpy()}\n')
o = tf.matmul(m, n)
print(o.numpy())

print('\n№10 Можно изменять типы данных тензоров функцией cast')
p = tf.cast(o, dtype=tf.int32)
print(p.numpy())

print('\n№11 Обработка форм тензора. Функция reshape позволяет преобразовывать тензоры в тензоры другой формы')
r = tf.ones(8,dtype=tf.int32)
print(f'{r.numpy()}\n')
s = tf.reshape(r, [2, 4])
print(f'{s.numpy()}\n')
t = tf.reshape(r, [2, 2, 2])
print(f'{t.numpy()}')

print('\n№12 Получение формы тензора с помощью функции get_shape()')
u = tf.ones((6, 3), dtype=tf.int32)
print(f'{u.numpy()}\n')
print(u.get_shape())

print('\n№13 Функция tf.expand_dims() добавляет в тензор размерность размера 1')
v = tf.expand_dims(u, 0)
print(f'{v.numpy()}\n')
print(v.get_shape())

print('\n№14 Функция tf.squeeze() удаляет из тензора все размерности размера 1')
w = tf.squeeze(v)
print(f'{w.numpy()}\n')
print(w.get_shape())

print('\n№15 Транслирование. Фактически просто можно складывать между собой тензоры разных размеров.' \
'При этом вектор, например, будет добавляться в каждую строку матрицы')
x = tf.ones((2, 2), dtype=tf.int32)
print(f'{x.numpy()}\n')
y = tf.range(0, 2, 1, dtype=tf.int32)
print(f'{y.numpy()}\n')
z = x + y
print(f'{z.numpy()}\n')
