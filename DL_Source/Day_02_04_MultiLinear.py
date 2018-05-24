# Day_02_04_MultiLinear.py
import tensorflow as tf
import numpy as np


def multi_linear_1():
    # hx = w1 * x1 + w2 * x2 + b
    #       1         1        0
    # y = x1 + x2
    x1 = [1, 0, 3, 0, 5]
    x2 = [0, 2, 0, 4, 0]
    y = [1, 2, 3, 4, 5]

    w1 = tf.Variable(tf.random_uniform([1]))
    w2 = tf.Variable(tf.random_uniform([1]))
    b = tf.Variable(tf.random_uniform([1]))

    hx = w1 * x1 + w2 * x2 + b
    cost = tf.reduce_mean((hx - y) ** 2)

    optimizer = tf.train.GradientDescentOptimizer(0.1)
    train = optimizer.minimize(cost)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(10):
        sess.run(train)
        print(i, sess.run(cost))

    sess.close()


def multi_linear_2():
    # 문제
    # x1과 x2를 하나의 변수로 합쳐보세요.
    # tf.matmul() 사용
    x = [[1., 0., 3., 0., 5.],
         [0., 2., 0., 4., 0.]]
    y = [1, 2, 3, 4, 5]

    w = tf.Variable(tf.random_uniform([1, 2]))
    b = tf.Variable(tf.random_uniform([1]))

    # hx = w[0] * x[0] + w[1] * x[1] + b
    hx = tf.matmul(w, x) + b
    cost = tf.reduce_mean((hx - y) ** 2)

    optimizer = tf.train.GradientDescentOptimizer(0.1)
    train = optimizer.minimize(cost)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(10):
        sess.run(train)
        print(i, sess.run(cost))

    sess.close()


def multi_linear_3():
    # 문제
    # bias를 없애보세요.
    # x = [[1., 0., 3., 0., 5.],
    #      [0., 2., 0., 4., 0.],
    #      [1., 1., 1., 1., 1.]]
    # x = [[1., 0., 3., 0., 5.],
    #      [1., 1., 1., 1., 1.],
    #      [0., 2., 0., 4., 0.]]
    x = [[1., 1., 1., 1., 1.],
         [1., 0., 3., 0., 5.],
         [0., 2., 0., 4., 0.]]
    y = [1, 2, 3, 4, 5]

    w = tf.Variable(tf.random_uniform([1, 3]))

    # hx = w[0] * x[0] + w[1] * x[1] + b * 1
    # hx = w[0] * x[0] + w[1] * x[1] + w[2] * x[2]
    # (1, 5) = (1, 3) x (3, 5)
    hx = tf.matmul(w, x)
    cost = tf.reduce_mean((hx - y) ** 2)

    optimizer = tf.train.GradientDescentOptimizer(0.1)
    train = optimizer.minimize(cost)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(100):
        sess.run(train)
        print(i, sess.run(cost))

    print(sess.run(w))
    sess.close()


def multi_linear_4():
    # 문제
    # 행렬 곱셈에서 w와 x의 순서를 바꿔보세요.
    # x = [[1., 1., 1., 1., 1.],
    #      [1., 0., 3., 0., 5.],
    #      [0., 2., 0., 4., 0.]]
    # x = np.transpose(x)
    # x = np.array(x, dtype=np.float32)
    # print(x.dtype)
    x = [[1., 1., 0.],
         [1., 0., 2.],
         [1., 3., 0.],
         [1., 0., 4.],
         [1., 5., 0.]]
    y = [[1], [2], [3], [4], [5]]

    w = tf.Variable(tf.random_uniform([3, 1]))

    # (5, 1) = (5, 3) x (3, 1)
    hx = tf.matmul(x, w)
    # hx = tf.matmul(x, w, True, True)      # 원래 데이터에 대해 동작.
    cost = tf.reduce_mean((hx - y) ** 2)

    optimizer = tf.train.GradientDescentOptimizer(0.1)
    train = optimizer.minimize(cost)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(10):
        sess.run(train)
        print(i, sess.run(cost))

    sess.close()


# multi_linear_1()
# multi_linear_2()
# multi_linear_3()
multi_linear_4()
