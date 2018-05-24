# Day_02_02_LinearRegression_cars.py
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def np_loadtxt():
    data = np.loadtxt('Data/simple.txt')
    print(data)
    print(type(data))   # <class 'numpy.ndarray'>
    print(data.shape, data.dtype)   # (3, 2) float64

    data = np.loadtxt('Data/simple2.txt',
                      skiprows=1,
                      delimiter=',',
                      dtype=np.int32)
    print(data)


# cars.csv 파일을 읽어서 반환하는 함수를 만드세요.
# 이때 반환값은 2개로 합니다. (x, y)
def get_cars():
    cars = np.loadtxt('Data/cars.csv',
                      delimiter=',')
    # print(cars)
    print(cars.shape)

    # speed, dist = [], []
    # # for i in cars:
    # #     print(i)
    #
    # for s, d in cars:
    #     # print(s, d)
    #     speed.append(s)
    #     dist.append(d)
    #
    # return speed, dist

    return cars[:, 0], cars[:, 1]


# 문제
# get_cars 함수 반환값을 텐서플로우 모델에 연동하세요.
def linear_regression_1():
    x, y = get_cars()

    w = tf.Variable(tf.random_uniform([1]))
    b = tf.Variable(tf.random_uniform([1]))

    hx = w * x + b

    cost = tf.reduce_mean((hx - y) ** 2)

    optimizer = tf.train.GradientDescentOptimizer(0.001)
    train = optimizer.minimize(cost)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(10):
        sess.run(train)
        print(i, sess.run(cost))

    sess.close()


# 문제
# 속도가 30과 50일 때의 제동거리를 예측해보세요. (placeholder 사용)
def linear_regression_2():
    xx, y = get_cars()

    x = tf.placeholder(tf.float32)

    w = tf.Variable(tf.random_uniform([1]))
    b = tf.Variable(tf.random_uniform([1]))

    hx = w * x + b

    cost = tf.reduce_mean((hx - y) ** 2)

    optimizer = tf.train.GradientDescentOptimizer(0.001)
    train = optimizer.minimize(cost)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(1000):
        sess.run(train, {x: xx})
        print(i, sess.run(cost, {x: xx}))
    print('-' * 50)

    print(sess.run(hx, {x: [30, 50]}))
    print(sess.run(hx, {x: 30}))

    y1 = sess.run(hx, {x: 0})
    y2 = sess.run(hx, {x: 30})
    y1, y2 = y1[0], y2[0]
    print(y2)

    pred = sess.run(hx, {x: [0, 30]})

    sess.close()

    # 문제
    # cars 데이터를 그래프에 출력하고
    # 우리가 학습한 결과(회귀선)를 표시해보세요.
    plt.plot(xx, y, 'ro')
    plt.plot([0, 30], [0, y2])
    plt.plot([0, 30], [y1, y2])
    plt.plot([0, 30], pred)
    plt.show()


# np_loadtxt()
# linear_regression_1()
linear_regression_2()

