# Day_02_03_cost_gradient_tensorflow.py
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def show_cost_1():
    # 문제
    # 파이썬으로 그렸던 cost 그래프를 텐서플로우 버전으로 그려보세요.
    x = [1, 2, 3]
    y = [1, 2, 3]

    w = tf.placeholder(tf.float32)

    hx = w * x
    cost = tf.reduce_mean((hx - y) ** 2)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # for i in np.linspace(-3, 5, 81):
    for i in np.arange(-3, 5, 0.1):
        c = sess.run(cost, {w: i})
        print('{:.1f} : {}'.format(i, c))

        plt.plot(i, c, 'ro')

    plt.show()
    sess.close()


def show_cost_2():
    # 문제
    # 반복문을 없애보세요.
    x = [1, 2, 3]
    y = [1, 2, 3]

    w = np.arange(-3, 5, 0.1).reshape(-1, 1)

    # (80, 3) = (80, 1) x (3,)
    hx = w * x
    print(type(hx))
    print(hx.shape)
    cost = tf.reduce_mean((hx - y) ** 2, axis=1)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    c = sess.run(cost)
    print(c)
    sess.close()

    plt.plot(w, c, 'ro')
    plt.show()


def show_cost_3():
    # 문제
    # 텐서플로우를 없애보세요. numpy만으로 구성해봅니다.
    x = [1, 2, 3]
    y = [1, 2, 3]

    w = np.arange(-3, 5, 0.1).reshape(-1, 1)
    hx = w * x
    c = np.mean((hx - y) ** 2, axis=1)

    print(c)

    plt.plot(w, c, 'ro')
    plt.show()


# show_cost_1()
# show_cost_2()
show_cost_3()
