import tensorflow as tf

tf.ones(1)
print('\n')


# Области имен
with tf.name_scope('name_of_the_scope'):
    print('Имена всех элементов созданных в рамках одной области имен будут дополняться префиксом названия этой области\n')


# Оптимизаторы
train_opt = tf.keras.optimizers.Adam(learning_rate=0.001)
print(f'{train_opt}\n')


# Прямое взятие градиентов
x = tf.Variable(-2.0)

with tf.GradientTape() as tape:
    y = x ** 2

df = tape.gradient(y, x)
tf.summary.scalar(name='df', data=df.numpy())
print(df.numpy())
