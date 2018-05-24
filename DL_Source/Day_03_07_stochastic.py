# Day_03_07_stochastic.py
import numpy as np
import matplotlib.pyplot as plt
import random


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def gradient_descent(x, y):
    m, n = x.shape              # 100, 3
    w = np.zeros([n, 1])        # (3, 1)
    lr = 0.01

    for _ in range(m):
        z = np.dot(x, w)        # (100, 1) = (100, 3) x (3, 1)
        h = sigmoid(z)
        e = h - y               # (100, 1)
        g = np.dot(x.T, e)      # (3, 1) = (3, 100) x (100, 1)
        w -= lr * g

    return w.reshape(-1)        # (3,)


def gradient_stoc(x, y):
    m, n = x.shape              # 100, 3
    w = np.zeros(n)             # (3,)
    lr = 0.01

    for i in range(m * 10):
        p = i % m
        z = np.sum(x[p] * w)    # scalar
        h = sigmoid(z)
        e = h - y[p]            # scalar
        g = x[p] * e            # (3,)
        w -= lr * g

    return w


def gradient_stoc_random(x, y):
    m, n = x.shape              # 100, 3
    w = np.zeros(n)             # (3,)
    lr = 0.01

    for _ in range(m * 10):
        p = random.randrange(m)
        z = np.sum(x[p] * w)    # scalar
        h = sigmoid(z)
        e = h - y[p]            # scalar
        g = x[p] * e            # (3,)
        w -= lr * g

    return w


def gradient_minibatch(x, y):
    m, n = x.shape              # 100, 3
    w = np.zeros([n, 1])        # (3, 1)
    lr = 0.01
    epochs = 10
    batch_size = 5
    iter = m // batch_size

    for _ in range(epochs):
        for i in range(iter):
            n1 = i * batch_size
            n2 = n1 + batch_size

            z = np.dot(x[n1:n2], w)    # (5, 1) = (5, 3) x (3, 1)
            h = sigmoid(z)
            e = h - y[n1:n2]           # (5, 1)
            g = np.dot(x[n1:n2].T, e)  # (3, 1) = (3, 5) x (5, 1)
            w -= lr * g

    return w.reshape(-1)                # (3,)


# 문제
# 아래 코드에 셔플 기능을 넣어주세요.
def gradient_minibatch_random(x, y):
    m, n = x.shape              # 100, 3
    w = np.zeros([n, 1])        # (3, 1)
    lr = 0.01
    epochs = 10
    batch_size = 5
    iter = m // batch_size

    action = np.hstack([x, y])

    for _ in range(epochs):
        x = action[:, :-1]
        y = action[:, -1:]

        for i in range(iter):
            n1 = i * batch_size
            n2 = n1 + batch_size

            z = np.dot(x[n1:n2], w)    # (5, 1) = (5, 3) x (3, 1)
            h = sigmoid(z)
            e = h - y[n1:n2]           # (5, 1)
            g = np.dot(x[n1:n2].T, e)  # (3, 1) = (3, 5) x (5, 1)
            w -= lr * g

        np.random.shuffle(action)

        # random.seed(_)
        # random.shuffle(x)
        # random.seed(_)
        # random.shuffle(y)

        # np.random.seed(_)
        # np.random.shuffle(x)
        # np.random.seed(_)
        # np.random.shuffle(y)

    return w.reshape(-1)                # (3,)


def decision_boundary(w, c):
    b, w1, w2 = w[0], w[1], w[2]

    y1 = -(w1 * -4 + b) / w2
    y2 = -(w1 *  4 + b) / w2

    plt.plot([-4, 4], [y1, y2], c)

    # hx = w1 * x1 + w2 * x2 + b
    # 0 = w1 * x1 + w2 * x2 + b
    # -(w1 * x1 + b) = w2 * x2
    # -(w1 * x1 + b) / w2 = x2


action = np.loadtxt('Data/action.txt', delimiter=',')
print(action.shape)

x = action[:, :-1]
y = action[:, -1:]
print(x.shape, y.shape)

# w = gradient_descent(x, y)
# decision_boundary(w, 'r')

for _, x1, x2, yy in action:
    plt.plot(x1, x2, 'ro' if yy else 'go')

decision_boundary(gradient_descent(x, y), 'r')
decision_boundary(gradient_stoc(x, y), 'g')
decision_boundary(gradient_stoc_random(x, y), 'b')
decision_boundary(gradient_minibatch(x, y), 'k')
decision_boundary(gradient_minibatch_random(x, y), 'm')

plt.show()
