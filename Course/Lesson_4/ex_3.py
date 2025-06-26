import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"

import tensorflow as tf

# №1. Если переменная одна, то в итоге и производная будет одна. Они будут суммированы (5+5=10)
x = tf.Variable(1.0)

with tf.GradientTape() as tape:
    # y = 2.0 * x ** 2 # получается производная равная 4
    # y = 3.0 * x ** 2 # получается производная равная 6
    y = [2.0, 3.0] * x ** 2 # поскольку переменная одна, то просто будет 4+6=10

df = tape.gradient(y, x)

print(f'{df}\n')


# №2. Если переменная - это вектор, то и производных будет 2
x1 = tf.Variable([1.0, 2.0])

with tf.GradientTape() as tape1:
    y1 = tf.reduce_sum([2.0, 3.0] * x1 ** 2)

df1 = tape1.gradient(y1, x1)

print(f'{df1}\n')


# №3. Все промежуточные манипуляции с переменными должны быть произведены внутри GradientTape() иначе производная вычисляться не будет
x3 = tf.Variable(2.0)
# z3 = x3 * 3 #если определить z3 здесь, то выдаст None. Нужно внутри GradientTape()

with tf.GradientTape() as tape3:
    z3 = x3 * 3
    y3 = z3 ** 2

df3 = tape3.gradient(y3, z3)

print(f'{df3}\n')

# №4. Если вычисляем производные в цикле, то изменение переменных простыми математическими действиями вне GradientTape() тоже превратит ее в константу.
# Нужно вместо этого использовать разные варианты функции assign

x4 = tf.Variable(3.0)

for i in range(2):
    with tf.GradientTape() as tape4:
        y4 = x4 + 1
    
    df4 = tape4.gradient(y4, x4)
    
    # x4 = x4 + 1 # так делать нельзя!!!!!!!!
    x4.assign_add(1.0)

    print(f'{df4}')

print('\n')

# №5. Внутри GradientTape() пользоваться только возможностями tensorflow!!!!! Например, если внутри tape ользоваться numpy для вычислений, то переведет в константу

# №6. Задавать значения переменных только в float типе!!!!!!!!! Иначе будут ошибки

# №7. Необходимо прописывать формулы в явном виде!!!!

x5 = tf.Variable(3.0)
w5 = tf.Variable(2.0)

with tf.GradientTape() as tape5:
    # w5 = w5.assign_add(x5) #если делать так, то будет None! Нужно делать явно
    # y5 = w5 ** 5
    q5 = w5 + x5
    y5 = q5 ** 5

df5 = tape5.gradient(y5, x5)

print(df5)
