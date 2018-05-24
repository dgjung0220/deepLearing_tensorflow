# Day_03_06_LogisticRegression_iris.py
import tensorflow as tf
import numpy as np
from sklearn import datasets, model_selection


def show_iris():
    iris = datasets.load_iris()
    # print(iris)
    print(type(iris))       # <class 'sklearn.utils.Bunch'>
    print(iris.keys())      # ['data', 'target', 'target_names', 'DESCR', 'feature_names']

    print(iris['feature_names'])
    # ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']

    print(iris['data'][:5])
    print(type(iris['data']))   # <class 'numpy.ndarray'>

    print(iris.target)
    print(iris.target_names)


# 문제
# setosa와 versicolor만 반환하는 함수를 만드세요.
def get_iris():
    iris = datasets.load_iris()

    x = iris.data[:100]
    y = iris.target[:100]

    x = np.float32(x)
    y = y.reshape(-1, 1)

    return x, y


def logistic_1():
    x, y = get_iris()
    print(x.shape, y.shape)

    # 문제
    # 로지스틱 리그레션 코드에 적용해보세요.
    # 행렬 곱셈에서 x를 앞쪽에 두겠습니다.
    w = tf.Variable(tf.random_uniform([4, 1], -1, 1))
    b = tf.Variable(tf.random_uniform([1], -1, 1))

    # (100, 1) = (100, 4) x (4, 1)
    z = tf.matmul(x, w) + b
    hx = tf.nn.sigmoid(z)
    cost = tf.reduce_mean(   y  * -tf.log(  hx) +
                          (1-y) * -tf.log(1-hx))

    optimizer = tf.train.GradientDescentOptimizer(0.1)
    train = optimizer.minimize(cost)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(10):
        sess.run(train)
        print(i, sess.run(cost))

    sess.close()


def get_train_test():
    x, y = get_iris()

    # 데이터 편중
    # x_train = x[:70]
    # x_test  = x[70:]
    # y_train = y[:70]
    # y_test  = y[70:]

    # x_train = x[15:-15]
    # # x_test  = np.array(list(x[:15]) + list(x[-15:]))
    # x_test  = np.vstack([x[:15], x[-15:]])
    # y_train = y[15:-15]
    # y_test  = np.vstack([y[:15], y[-15:]])

    np.random.seed(1)
    np.random.shuffle(x)
    np.random.seed(1)
    np.random.shuffle(y)

    x_train = x[:70]
    x_test  = x[70:]
    y_train = y[:70]
    y_test  = y[70:]

    return x_train, x_test, y_train, y_test


# 문제
# train과 test 데이터셋을 7:3으로 반환하는
# get_train_test 함수를 만들고
# test 셋에 대해서 결과를 예측해보세요.
# 반환값 4개 : x_train, x_test, y_train, y_test.
def logistic_2():
    x_train, x_test, y_train, y_test = get_train_test()

    x = tf.placeholder(tf.float32)

    w = tf.Variable(tf.random_uniform([4, 1], -1, 1))
    b = tf.Variable(tf.random_uniform([1], -1, 1))

    # (70, 1) = (70, 4) x (4, 1)
    z = tf.matmul(x, w) + b
    hx = tf.nn.sigmoid(z)
    cost = tf.reduce_mean(y_train * -tf.log(hx) +
                          (1 - y_train) * -tf.log(1 - hx))

    optimizer = tf.train.GradientDescentOptimizer(0.1)
    train = optimizer.minimize(cost)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(100):
        sess.run(train, {x: x_train})
        print(i, sess.run(cost, {x: x_train}))
    print('-' * 50)

    pred = sess.run(hx, {x: x_test})
    pred = pred.reshape(-1)
    y_test = y_test.reshape(-1)
    print(pred)
    print(y_test)

    pred_bool = (pred >= 0.5)
    print(pred_bool)

    equals = (pred_bool == y_test)
    print(equals)
    print('acc :', np.mean(equals))

    sess.close()


def logistic_3():
    iris = datasets.load_iris()
    # data = model_selection.train_test_split(
    #            iris.data[:100],
    #            iris.target[:100].reshape(-1, 1))

    # data = model_selection.train_test_split(
    #            iris.data[:100],
    #            iris.target[:100].reshape(-1, 1),
    #            shuffle=True,
    #            train_size=70)

    data = model_selection.train_test_split(
               iris.data[:100],
               iris.target[:100].reshape(-1, 1),
               test_size=0.3)

    x_train, x_test, y_train, y_test = data
    print(x_train.shape, x_test.shape)
    print(y_train.shape, y_test.shape)

    x = tf.placeholder(tf.float32)

    w = tf.Variable(tf.random_uniform([4, 1], -1, 1))
    b = tf.Variable(tf.random_uniform([1], -1, 1))

    # (70, 1) = (70, 4) x (4, 1)
    z = tf.matmul(x, w) + b
    hx = tf.nn.sigmoid(z)
    cost = tf.reduce_mean(y_train * -tf.log(hx) +
                          (1 - y_train) * -tf.log(1 - hx))

    optimizer = tf.train.GradientDescentOptimizer(0.1)
    train = optimizer.minimize(cost)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(100):
        sess.run(train, {x: x_train})

    pred = sess.run(hx, {x: x_test})
    pred = pred.reshape(-1)
    y_test = y_test.reshape(-1)

    pred_bool = (pred >= 0.5)

    equals = (pred_bool == y_test)
    print('acc :', np.mean(equals))
    print('acc :', equals.mean())

    sess.close()


# show_iris()
# logistic_1()
# logistic_2()
logistic_3()




print('\n\n\n\n\n\n\n')