# Day_04_02_softmax.py
import math
import tensorflow as tf
import numpy as np

np.set_printoptions(linewidth=1000)


def softmax():
    a = math.e ** 2.0
    b = math.e ** 1.0
    c = math.e ** 0.1

    base = a + b + c

    print(a / base)     # 0.6590011388859678
    print(b / base)     # 0.2424329707047139
    print(c / base)     # 0.09856589040931818


# 문제
# softmax.txt 파일을 가져와서
# 아래 코드가 동작하도록 수정해보세요.
def cross_entropy_1():
    data = np.loadtxt('Data/softmax.txt',
                      dtype=np.float32)

    x = data[:, :3]     # (8, 3)
    y = data[:, 3:]     # (8, 3)

    w = tf.Variable(tf.random_uniform([3, 3], -1, 1))

    # (8, 3) = (8, 3) x (3, 3)
    z = tf.matmul(x, w)
    hx = tf.nn.softmax(z)
    cost_i = tf.nn.softmax_cross_entropy_with_logits(logits=z,
                                                     labels=y)
    cost = tf.reduce_mean(cost_i)

    optimizer = tf.train.GradientDescentOptimizer(0.1)
    train = optimizer.minimize(cost)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(10):
        sess.run(train)
        print(i, sess.run(cost))

    sess.close()


# 문제
# 행렬 곱셈에서 x와 w의 위치를 바꿔보세요.
def cross_entropy_2():
    data = np.loadtxt('Data/softmax.txt',
                      dtype=np.float32,
                      unpack=True)
    print(data.shape)

    x = data[:3]     # (3, 8)
    y = data[3:]     # (3, 8)

    w = tf.Variable(tf.zeros([3, 3]))

    # (3, 8) = (3, 3) x (3, 8)
    z = tf.matmul(w, x)
    # (3, 8)
    hx = tf.nn.softmax(z, axis=0)
    # (8,)
    cost_i = tf.nn.softmax_cross_entropy_with_logits(logits=z,
                                                     labels=y,
                                                     dim=0)
    # scalar
    cost = tf.reduce_mean(cost_i)

    optimizer = tf.train.GradientDescentOptimizer(0.1)
    train = optimizer.minimize(cost)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    print('w :\n', sess.run(w))
    print('z :\n', sess.run(z))
    print('hx :\n', sess.run(hx))
    print('cost_i :\n', sess.run(cost_i))
    print('cost :\n', sess.run(cost))

    # for i in range(10):
    #     sess.run(train)
    #     print(i, sess.run(cost))

    sess.close()


# softmax()
# cross_entropy_1()
cross_entropy_2()
