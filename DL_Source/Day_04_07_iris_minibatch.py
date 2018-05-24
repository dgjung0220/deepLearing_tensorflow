# Day_04_07_iris_minibatch.py
import tensorflow as tf
import numpy as np
from sklearn import model_selection


data = np.loadtxt('Data/iris_softmax.csv',
                  delimiter=',',
                  dtype=np.float32)

data = model_selection.train_test_split(data[:, :-3],
                                        data[:, -3:],
                                        train_size=120)
x_train, x_test, y_train, y_test = data

x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

w = tf.Variable(tf.random_uniform([5, 3], -1, 1))

z = tf.matmul(x, w)
hx = tf.nn.softmax(z)
cost_i = tf.reduce_sum(y * -tf.log(hx), axis=1)
cost = tf.reduce_mean(cost_i)

optimizer = tf.train.GradientDescentOptimizer(0.1)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# 문제
# 미니배치 방식으로 코드를 수정해주세요.
# epochs는 15회, batch_size는 5개로 합니다.
epochs = 15
batch_size = 5
iter = len(x_train) // batch_size
print(iter)

# for i in range(epochs):
#     loss = 0
#     for j in range(iter):
#         n1 = j * batch_size
#         n2 = n1 + batch_size
#
#         sess.run(train, {x: x_train[n1:n2],
#                          y: y_train[n1:n2]})
#         loss += sess.run(cost, {x: x_train[n1:n2],
#                                 y: y_train[n1:n2]})
#     print(i, loss / iter)

for i in range(epochs):
    for j in range(iter):
        n1 = j * batch_size
        n2 = n1 + batch_size

        sess.run(train, {x: x_train[n1:n2],
                         y: y_train[n1:n2]})
    print(i, sess.run(cost, {x: x_train,
                             y: y_train}))

# for i in range(epochs):
#     total = 0
#     for j in range(iter):
#         n1 = j * batch_size
#         n2 = n1 + batch_size
#
#         _, loss = sess.run([train, cost], {x: x_train[n1:n2],
#                                            y: y_train[n1:n2]})
#         total += loss
#
#     print(i, total / iter)

sess.close()












