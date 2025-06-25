import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

tf.zeros(1)
print('\n')


# Создаем игрушечную выборку данных
N = 100

x_zeros = np.random.multivariate_normal(mean=(-1, -1), cov=0.2*np.eye(2), size=int(N/2))
x_ones = np.random.multivariate_normal(mean=(1, 1), cov=0.2*np.eye(2), size=int(N/2))
y_zeros = np.zeros(int(N/2))
y_ones = np.ones(int(N/2))
x = np.vstack([x_zeros,x_ones]).transpose()[0]
y = np.vstack([x_zeros,x_ones]).transpose()[1]

# plt.scatter(x=x, y=y, c=np.concatenate([y_zeros, y_ones]))
# plt.show()

# Тренируем модель
EPOCHS = 500
learning_rate = 0.02

opt = tf.keras.optimizers.sigmoid(learning_rate)



# НЕ НАШЕЛ!!!!!!!! 