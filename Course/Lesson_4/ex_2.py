import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"

import tensorflow as tf

# Автоматическое дифференциирование. Нюансы

#Объект GradientTape вычисляет производные только для переменных.
# Для констант производная вычисляться не будет
w = tf.Variable(tf.random.normal([3, 2]))
b = tf.Variable(tf.zeros(2, dtype=tf.float32))
# b = tf.constant(tf.zeros(2, dtype=tf.float32)) #от константы производная вычисляьбся не будет
# b = tf.Variable(tf.zeros(2, dtype=tf.float32), trainable=False) #trainable = False запрещает вычислять производные по конкретной переменной
x = tf.Variable([[-2.0, 1.0, 3.0]])


# with tf.GradientTape(watch_accessed_variables=False) as tape: #запрещает вычислять производные по переменным.
# Разрешить можно в теле tape по отдельным переменным с помощью tape.watch(b)

# with tf.GradientTape(persistent=True) as tape: #не освобождает резурсы после вычисления производной и его можно вызывать несколько раз подряд для разных функций
# После вычисления всех производных таким образом нужно явно удалить объект tape с помощью del tape, чтобы ресурсы освободились.

with tf.GradientTape() as tape:
    # tape.watch(b) #разрешает вычислять производные по переменной, если запрещено по всем
    y = x @ w + b
    loss = tf.reduce_min(y ** 2)

df = tape.gradient(loss, [x, b])

# del tape #принудительное удаление tape, если было запрещено освобождение ресурсов при его создании

print(df[0], df[1], sep='\n\n')
