import os
import numpy as np
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Игрушечный набор данных для регресии
N = 100
w_true = 5
b_true = 2
x_np = np.random.uniform(low=0, high=1, size=[N, 1])
noise = np.random.normal(loc=0, scale=0.1, size=[N, 1])
y_np = w_true * x_np + b_true + noise

plt.scatter(x=x_np, y=y_np)
plt.show()


# Игрушечный набор данных для классификации
N = 100

x_zeros = np.random.multivariate_normal(mean=(-1, -1), cov=0.2*np.eye(2), size=int(N/2))
x_ones = np.random.multivariate_normal(mean=(1, 1), cov=0.2*np.eye(2), size=int(N/2))
y_zeros = np.zeros(int(N/2))
y_ones = np.ones(int(N/2))

plt.scatter(x=np.vstack([x_zeros,x_ones]).transpose()[0], y=np.vstack([x_zeros,x_ones]).transpose()[1], c=np.concatenate([y_zeros, y_ones]))
plt.show()
