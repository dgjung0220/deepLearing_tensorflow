# Day_04_05_ensemble.py
import numpy as np
import tensorflow as tf
from sklearn import model_selection


def make_prediction(x_train, x_test, y_train):
    x = tf.placeholder(tf.float32)
    w = tf.Variable(tf.random_uniform([5, 3], -1, 1))

    z = tf.matmul(x, w)
    hx = tf.nn.softmax(z)
    cost_i = tf.reduce_sum(y_train * -tf.log(hx), axis=1)
    cost = tf.reduce_mean(cost_i)

    optimizer = tf.train.GradientDescentOptimizer(0.1)
    train = optimizer.minimize(cost)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(150):
        sess.run(train, {x: x_train})
        # print(i, sess.run(cost, {x: x_train}))

    pred = sess.run(hx, {x: x_test})
    sess.close()

    return pred


def show_accuracy(pred, y_test):
    pred_arg = np.argmax(pred, axis=1)
    test_arg = np.argmax(y_test, axis=1)

    print('acc :', np.mean(pred_arg == test_arg))


data = np.loadtxt('Data/iris_softmax.csv',
                  delimiter=',',
                  dtype=np.float32)

data = model_selection.train_test_split(data[:, :-3],
                                        data[:, -3:],
                                        train_size=120)
x_train, x_test, y_train, y_test = data

total = np.zeros(y_test.shape)
for i in range(7):
    pred = make_prediction(x_train, x_test, y_train)
    show_accuracy(pred, y_test)

    total += pred

# 문제
# total의 정확도를 알려주세요.
print('-' * 50)
show_accuracy(total, y_test)

print(total[:3])

