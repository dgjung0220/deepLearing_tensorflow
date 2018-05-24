# Day_04_03_softmax_iris.py
import numpy as np
import csv
import tensorflow as tf
import pandas as pd
from sklearn import model_selection, preprocessing


def make_iris_softmax():
    def make_onehot(sp):
        if sp == 'setosa': return [1, 0, 0]
        if sp == 'versicolor': return [0, 1, 0]
        return [0, 0, 1]

    f = open('Data/iris.csv', 'r', encoding='utf-8')

    # skip header.
    f.readline()

    iris = []
    for row in csv.reader(f):
        # print(row)

        item = [1.0]
        item += [float(i) for i in row[1:-1]]
        item += make_onehot(row[-1])
        # print(item)

        iris.append(item)

    f.close()

    f = open('Data/iris_softmax.csv', 'w', encoding='utf-8', newline='')

    csv.writer(f).writerows(iris)
    # csv.writer(f, delimiter=' ').writerows(iris)
    f.close()


# 문제
# iris_softmax.csv 파일을 읽어서
# 120개의 데이터로 학습하고 30개의 데이터로 정확도를 검증해보세요.
def softmax_1():
    data = np.loadtxt('Data/iris_softmax.csv',
                      delimiter=',',
                      dtype=np.float32)
    print(data.shape)

    # np.random.shuffle(data)
    #
    # x_train, x_test = data[:120, :-3], data[120:, :-3]
    # y_train, y_test = data[:120, -3:], data[120:, -3:]

    data = model_selection.train_test_split(data[:, :-3],
                                            data[:, -3:],
                                            train_size=120)
    x_train, x_test, y_train, y_test = data

    print(x_train.shape, x_test.shape)
    print(y_train.shape, y_test.shape)

    x = tf.placeholder(tf.float32, shape=[None, 5])
    w = tf.Variable(tf.random_uniform([5, 3], -1, 1))

    # (120, 3) = (120, 5) x (5, 3)
    z = tf.matmul(x, w)
    hx = tf.nn.softmax(z)
    # cost_i = tf.reduce_sum(y_train * -tf.log(hx), axis=1)
    cost_i = tf.nn.softmax_cross_entropy_with_logits(logits=z,
                                                     labels=y_train)
    cost = tf.reduce_mean(cost_i)

    optimizer = tf.train.GradientDescentOptimizer(0.1)
    train = optimizer.minimize(cost)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(1000):
        sess.run(train, {x: x_train})
        print(i, sess.run(cost, {x: x_train}))
    print('-' * 50)

    pred = sess.run(hx, {x: x_test})
    # pred = sess.run(z, {x: x_test})
    print(pred[:3])
    print(y_test[:3])

    pred_arg = np.argmax(pred, axis=1)
    test_arg = np.argmax(y_test, axis=1)
    print(pred_arg)
    print(test_arg)

    print('acc :', np.mean(pred_arg == test_arg))
    sess.close()


def softmax_2():
    iris = pd.read_csv('Data/iris.csv',
                       index_col=0)
    print(iris.shape)
    print(iris)

    xx = iris.values[:, :-1]
    print(xx[:3])

    yy = iris.values[:, -1:]
    print(yy[:3])

    # 문제
    # xx와 yy에 전처리 작업을 해보세요.
    xx = preprocessing.add_dummy_feature(xx)
    yy = preprocessing.LabelBinarizer().fit_transform(yy)

    data = model_selection.train_test_split(xx,
                                            yy,
                                            train_size=120)
    x_train, x_test, y_train, y_test = data

    x = tf.placeholder(tf.float32)
    w = tf.Variable(tf.random_uniform([5, 3], -1, 1))

    # (120, 3) = (120, 5) x (5, 3)
    z = tf.matmul(x, w)
    hx = tf.nn.softmax(z)
    # cost_i = tf.reduce_sum(y_train * -tf.log(hx), axis=1)
    cost_i = tf.nn.softmax_cross_entropy_with_logits(logits=z,
                                                     labels=y_train)
    cost = tf.reduce_mean(cost_i)

    optimizer = tf.train.GradientDescentOptimizer(0.1)
    train = optimizer.minimize(cost)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(1000):
        sess.run(train, {x: x_train})
        print(i, sess.run(cost, {x: x_train}))
    print('-' * 50)

    pred = sess.run(hx, {x: x_test})
    # pred = sess.run(z, {x: x_test})
    print(pred[:3])
    print(y_test[:3])

    pred_arg = np.argmax(pred, axis=1)
    test_arg = np.argmax(y_test, axis=1)
    print(pred_arg)
    print(test_arg)

    print('acc :', np.mean(pred_arg == test_arg))
    sess.close()


def softmax_sparse():
    iris = pd.read_csv('Data/iris.csv',
                       index_col=0)
    xx = iris.values[:, :-1]
    yy = iris.values[:, -1:]

    xx = preprocessing.add_dummy_feature(xx)
    # yy = preprocessing.LabelBinarizer().fit_transform(yy)
    yy = preprocessing.LabelEncoder().fit_transform(yy)

    data = model_selection.train_test_split(xx,
                                            yy,
                                            train_size=120)
    x_train, x_test, y_train, y_test = data

    x = tf.placeholder(tf.float32)
    w = tf.Variable(tf.random_uniform([5, 3], -1, 1))

    # (120, 3) = (120, 5) x (5, 3)
    z = tf.matmul(x, w)
    hx = tf.nn.softmax(z)
    cost_i = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=z,
                                                     labels=y_train)
    # cost_i = tf.nn.softmax_cross_entropy_with_logits(logits=z,
    #                                                  labels=y_train)
    cost = tf.reduce_mean(cost_i)

    optimizer = tf.train.GradientDescentOptimizer(0.1)
    train = optimizer.minimize(cost)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(1000):
        sess.run(train, {x: x_train})
        print(i, sess.run(cost, {x: x_train}))
    print('-' * 50)

    pred = sess.run(hx, {x: x_test})
    print(pred[:3])
    print(y_test[:3])

    pred_arg = np.argmax(pred, axis=1)
    test_arg = y_test  # np.argmax(y_test, axis=1)
    print(pred_arg)

    print('acc :', np.mean(pred_arg == test_arg))
    sess.close()


# make_iris_softmax()
# softmax_1()
# softmax_2()
softmax_sparse()



print('\n\n\n\n\n\n')

