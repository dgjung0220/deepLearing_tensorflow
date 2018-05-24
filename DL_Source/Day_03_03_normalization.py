# Day_03_03_normalization.py
import tensorflow as tf
import numpy as np
# import sklearn
from sklearn import preprocessing


def minmax(data):
    # (나 - 최소) / (최대 - 최소)
    mx = np.max(data, axis=0)
    mn = np.min(data, axis=0)
    rn = mx - mn
    return (data - mn) / rn


def minmax_2(data):
    scaler = preprocessing.MinMaxScaler()
    scaler.fit(data)
    return scaler.transform(data)


def test_normalization(use=False):
    data = [[828, 833, 1908100, 928, 812],
            [829, 823, 1928100, 828, 822],
            [820, 843, 1908300, 728, 832],
            [821, 851, 1909100, 628, 842],
            [822, 836, 1901100, 528, 852],
            [823, 837, 1900100, 428, 862],
            [824, 839, 1903100, 328, 872]]
    data = np.array(data, dtype=np.float32)

    if use:
        # data = minmax(data)
        data = minmax_2(data)

    # x = data[:, :-1].transpose()
    x = data[:, :-1].T
    y = data[:, -1]

    w = tf.Variable(tf.random_uniform([1, 4]))

    # (1, 7) = (1, 4) x (4, 7)
    hx = tf.matmul(w, x)
    cost = tf.reduce_mean((hx - y) ** 2)

    optimizer = tf.train.GradientDescentOptimizer(0.1)
    train = optimizer.minimize(cost)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(10):
        sess.run(train)
        print(i, sess.run(cost))

    sess.close()


# test_normalization(False)
test_normalization(True)



