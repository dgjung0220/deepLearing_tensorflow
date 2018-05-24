# Day_05_01_save_model.py
import tensorflow as tf
import numpy as np


def save_model_tf():
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

    saver = tf.train.Saver()

    for i in range(101):
        sess.run(train, {x: xx})
        # saver.save(sess, 'model/second', global_step=i)

        if i % 10 == 0:
            print(i, sess.run(cost, {x: xx}))
            saver.save(sess, 'model/third', global_step=i)

    # saver.save(sess, 'model/first')

    # [4.899799  6.8263044]
    print(sess.run(hx, {x: [5, 7]}))
    sess.close()


def restore_model_tf_1():
    w = tf.Variable(tf.random_uniform([1]))
    b = tf.Variable(tf.random_uniform([1]))

    x = tf.placeholder(tf.float32)
    hx = w * x + b

    sess = tf.Session()

    latest = tf.train.latest_checkpoint('model')
    print(latest)

    saver = tf.train.Saver()
    saver.restore(sess, latest)

    # [4.899799  6.8263044]
    print(sess.run(hx, {x: [5, 7]}))
    sess.close()


def restore_model_tf_2():
    w = tf.Variable([0.])
    b = tf.Variable([0.])

    latest = tf.train.latest_checkpoint('model')
    print(latest)

    saver = tf.train.Saver()

    sess = tf.Session()
    saver.restore(sess, latest)

    hx = w * [5, 7] + b

    # [4.899799  6.8263044]
    print(sess.run(hx))
    print(sess.run(w * [5, 7] + b))
    sess.close()


def save_model_text():
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

    for i in range(101):
        sess.run(train, {x: xx})

    ww, bb = sess.run([w, b])
    print(ww, bb)

    # [4.9642525 6.9380326]
    print(sess.run(hx, {x: [5, 7]}))
    sess.close()

    # -------------------- #

    f = open('model/anywhere.txt', 'w', encoding='utf-8')
    f.write('{},{}'.format(ww[0], bb[0]))
    f.close()


def restore_model_text():
    f = open('model/anywhere.txt', 'r', encoding='utf-8')
    text = f.readline()
    f.close()

    print(text)

    w, b = [float(i) for i in text.split(',')]
    print(w, b)

    # [4.96425229 6.93803233] 살짝 오차 발생.
    hx = w * np.array([5, 7]) + b
    print(hx)


# save_model_tf()
# restore_model_tf_1()
# restore_model_tf_2()

# save_model_text()
restore_model_text()


