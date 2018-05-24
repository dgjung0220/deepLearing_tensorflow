# Day_04_06_ensemble_class.py
import numpy as np
import tensorflow as tf
from sklearn import model_selection


class Single:
    def __init__(self, x_train, x_test, y_train, y_test,
                 lr=0.1, loop_count=100, verbose=False):
        n_features = x_train.shape[-1]
        n_classes = y_train.shape[-1]

        x = tf.placeholder(tf.float32)
        w = tf.Variable(tf.random_uniform([n_features, n_classes],
                                          -1, 1))

        z = tf.matmul(x, w)
        hx = tf.nn.softmax(z)
        cost_i = tf.reduce_sum(y_train * -tf.log(hx), axis=1)
        cost = tf.reduce_mean(cost_i)

        optimizer = tf.train.GradientDescentOptimizer(lr)
        train = optimizer.minimize(cost)

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        for i in range(loop_count):
            sess.run(train, {x: x_train})

            if verbose:
                print(i, sess.run(cost, {x: x_train}))

        self.pred = sess.run(hx, {x: x_test})
        self.test = y_test
        sess.close()

    def show_accuracy(self):
        pred_arg = np.argmax(self.pred, axis=1)
        test_arg = np.argmax(self.test, axis=1)

        print('acc :', np.mean(pred_arg == test_arg))


class Ensemble:
    def __init__(self, count, x_train, x_test, y_train, y_test,
                 lr=0.1, loop_count=100):
        self.models = [Single(x_train, x_test, y_train, y_test,
                              lr, loop_count)
                       for _ in range(count)]
        self.test = y_test

    def show_accuracy(self):
        total = np.zeros_like(self.test)
        for m in self.models:
            m.show_accuracy()
            total += m.pred

        print('-' * 50)
        pred_arg = np.argmax(total, axis=1)
        test_arg = np.argmax(self.test, axis=1)

        print('acc :', np.mean(pred_arg == test_arg))


data = np.loadtxt('Data/iris_softmax.csv',
                  delimiter=',',
                  dtype=np.float32)

data = model_selection.train_test_split(data[:, :-3],
                                        data[:, -3:],
                                        train_size=120)
x_train, x_test, y_train, y_test = data

# s = Single(x_train, x_test, y_train, y_test)
# s.show_accuracy()

e = Ensemble(7, x_train, x_test, y_train, y_test, loop_count=150)
e.show_accuracy()
