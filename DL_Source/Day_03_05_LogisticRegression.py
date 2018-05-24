# Day_03_05_LogisticRegression.py
import tensorflow as tf
import numpy as np

x = [[1., 1., 1., 1., 1., 1.],
     [2., 3., 3., 5., 7., 2.],
     [1., 2., 5., 5., 5., 6.]]
y = [0, 0, 0, 1, 1, 1]
y = np.array(y)

w = tf.Variable(tf.random_uniform([1, 3], -1, 1))

z = tf.matmul(w, x)
# hx = 1 / (1 + tf.exp(-z))
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








