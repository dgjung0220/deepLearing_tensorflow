# Day_02_05_MultiLinear_trees.py
import numpy as np
import tensorflow as tf

# scikit-


def get_trees():
    trees = np.loadtxt('Data/trees.csv',
                       skiprows=1,
                       delimiter=',',
                       dtype=np.float32)
    x = trees[:, :2]
    y = trees[:, 2:]
    print(x.shape, y.shape)     # (31, 2) (31, 1)

    return x, y

# 106번으로 연결하세요.

# 문제
# trees.csv 파일을 가져와서 처리해주세요.
# x는 Girth, Height
# y는 Volume으로 하겠습니다.
# Girth와 Height가 (12, 75)일 때와
# Girth와 Height가 (15, 80)일 때의 Volume을 알고 싶습니다.
def trees_1():
    xx, y = get_trees()

    x = tf.placeholder(tf.float32)

    w = tf.Variable(tf.random_uniform([2, 1]))
    b = tf.Variable(tf.random_uniform([1]))

    # (31, 1) = (31, 2) x (2, 1)
    hx = tf.matmul(x, w) + b
    cost = tf.reduce_mean((hx - y) ** 2)

    optimizer = tf.train.GradientDescentOptimizer(0.0001)
    train = optimizer.minimize(cost)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(10):
        sess.run(train, {x: xx})
        print(i, sess.run(cost, {x: xx}))
    print('-' * 50)

    # Girth와 Height가 (12, 75)일 때와
    # Girth와 Height가 (15, 80)일 때의 Volume을 알고 싶습니다.
    print(sess.run(hx, {x: xx}))

    print(xx)

    # [[ 8.3 70. ]
    #  [ 8.6 65. ]
    #  ...
    #  [18.  80. ]
    #  [20.6 87. ]]

    # [[12, 75]]

    # print(sess.run(hx, {x: [12, 75]}))        # error.
    print(sess.run(hx, {x: [[12, 75]]}))
    print(sess.run(hx, {x: [[12, 75], (15, 80)]}))

    sess.close()


def trees_2():
    xx, y = get_trees()

    rows = []
    for row in xx:
        rows.append([1.] + list(row))   # extend()
        # print(rows[-1])
    xx = rows

    x = tf.placeholder(tf.float32)

    w = tf.Variable(tf.random_uniform([3, 1]))

    # (31, 1) = (31, 3) x (3, 1)
    hx = tf.matmul(x, w)
    cost = tf.reduce_mean((hx - y) ** 2)

    optimizer = tf.train.GradientDescentOptimizer(0.0001)
    train = optimizer.minimize(cost)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(10):
        sess.run(train, {x: xx})
        print(i, sess.run(cost, {x: xx}))
    print('-' * 50)

    print(sess.run(hx, {x: [[1, 12, 75],
                            (1, 15, 80)]}))
    sess.close()


# trees_1()
trees_2()




