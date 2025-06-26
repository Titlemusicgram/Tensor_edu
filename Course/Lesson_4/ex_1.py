import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"

import tensorflow as tf

# Автоматическое дифференциирование. В общем случае

# Производная
print('Производная\n')
x = tf.Variable(-2.0)

with tf.GradientTape() as tape:
    y = x ** 2

df = tape.gradient(y, x)
print(f'{df.numpy()}\n')

