# Day_01_04_LinearRegressio.py
import tensorflow as tf


def basic():
    a = tf.constant(3)
    b = tf.Variable(7)

    print(a)
    print(b)

    sess = tf.Session()

    # sess.run(tf.global_variables_initializer())
    sess.run(b.initializer)

    print(sess.run(a))
    print(sess.run(b))

    sess.close()


def linear_regression_1():
    x = [1, 2, 3]
    y = [1, 2, 3]

    # w = tf.Variable(10.)
    # b = tf.Variable(20.)
    w = tf.Variable(tf.random_uniform([1]))
    b = tf.Variable(tf.random_uniform([1]))

    # hx = wx + b
    # hx = tf.add(tf.multiply(w, x), b)
    hx = w * x + b

    # c += (hx - y[i]) ** 2
    # cost = tf.reduce_mean(tf.square(hx - y))
    cost = tf.reduce_mean((hx - y) ** 2)

    # optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
    # train = optimizer.minimize(loss=cost)
    optimizer = tf.train.GradientDescentOptimizer(0.1)
    train = optimizer.minimize(cost)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(1000):
        # print(sess.run(train))
        sess.run(train)
        print(i, sess.run(cost))

    # 문제
    # x가 5와 7일 때의 y값을 예측해보세요.
    ww = sess.run(w)
    bb = sess.run(b)

    print('5 :', ww * 5 + bb)
    print('7 :', ww * 7 + bb)
    print(ww * [5, 7] + bb)

    print(ww, bb)
    sess.close()


def linear_regression_2():
    xx = [1, 2, 3]
    y = [1, 2, 3]

    x = tf.placeholder(tf.float32)

    w = tf.Variable(tf.random_uniform([1]))
    b = tf.Variable(tf.random_uniform([1]))

    hx = w * x + b

    cost = tf.reduce_mean((hx - y) ** 2)

    optimizer = tf.train.GradientDescentOptimizer(0.1)
    train = optimizer.minimize(cost)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(10):
        sess.run(train, feed_dict={x: xx})
        print(i, sess.run(cost, {x: xx}))
    print('-' * 50)

    # 문제
    # placeholder 변수인 x를 사용해서 5와 7일 때의 결과를 예측해보세요.
    print(sess.run(hx, {x: 5}))
    print(sess.run(hx, {x: 7}))
    print(sess.run(hx, {x: xx}))
    print(sess.run(hx, {x: [1, 2, 3]}))
    print(sess.run(hx, {x: [5, 7]}))
    sess.close()


def linear_regression_3():
    xx = [1, 2, 3]
    y = [1, 2, 3]

    x = tf.placeholder(tf.float32)

    w = tf.Variable(tf.random_uniform([1]))
    b = tf.Variable(tf.random_uniform([1]))

    hx = w * x + b

    cost = tf.reduce_mean((hx - y) ** 2)

    optimizer = tf.train.GradientDescentOptimizer(0.1)
    train = optimizer.minimize(cost)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(2001):
        sess.run(train, feed_dict={x: xx})

        if i % 20 == 0:
            print(i, sess.run(cost, {x: xx}))


# basic()
# linear_regression_1()
# linear_regression_2()
linear_regression_3()
