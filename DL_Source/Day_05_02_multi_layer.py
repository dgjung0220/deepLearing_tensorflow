# Day_05_02_multi_layer.py
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


def show_accuracy(hx, sess, x, y, keep_rate, prompt, dataset):
    pred = tf.equal(tf.argmax(hx, axis=1), tf.argmax(y, axis=1))
    acc = tf.reduce_mean(tf.cast(pred, tf.float32))

    feed = {x: dataset.images,
            y: dataset.labels,
            keep_rate: 1.0}
    print(prompt, ':', sess.run(acc, feed))


def softmax(x, y, _):
    w = tf.Variable(tf.zeros([784, 10]))  # 784 = 28 * 28
    b = tf.Variable(tf.zeros([10]))

    z = tf.matmul(x, w) + b
    hx = tf.nn.softmax(z)

    cost_i = tf.nn.softmax_cross_entropy_with_logits(logits=z,
                                                     labels=y)
    cost = tf.reduce_mean(cost_i)

    return hx, cost


def multi_relu_1(x, y, _):
    w1 = tf.Variable(tf.random_normal([784, 256]))
    w2 = tf.Variable(tf.random_normal([256, 256]))
    w3 = tf.Variable(tf.random_normal([256, 10]))

    b1 = tf.Variable(tf.random_normal([256]))
    b2 = tf.Variable(tf.random_normal([256]))
    b3 = tf.Variable(tf.random_normal([10]))

    z1 = tf.matmul(x, w1) + b1
    r1 = tf.nn.relu(z1)

    z2 = tf.matmul(r1, w2) + b2
    r2 = tf.nn.relu(z2)

    z3 = tf.matmul(r2, w3) + b3
    # hx = tf.nn.softmax(z3)
    cost_i = tf.nn.softmax_cross_entropy_with_logits(logits=z3,
                                                     labels=y)
    cost = tf.reduce_mean(cost_i)

    return z3, cost


def multi_relu_2(x, y, _):
    w1 = tf.Variable(tf.random_normal([784, 256]))
    w2 = tf.Variable(tf.random_normal([256, 256]))
    w3 = tf.Variable(tf.random_normal([256, 10]))

    b1 = tf.Variable(tf.zeros([256]))
    b2 = tf.Variable(tf.zeros([256]))
    b3 = tf.Variable(tf.zeros([10]))

    z1 = tf.matmul(x, w1) + b1
    r1 = tf.nn.relu(z1)

    z2 = tf.matmul(r1, w2) + b2
    r2 = tf.nn.relu(z2)

    z3 = tf.matmul(r2, w3) + b3
    # hx = tf.nn.softmax(z3)
    cost_i = tf.nn.softmax_cross_entropy_with_logits(logits=z3,
                                                     labels=y)
    cost = tf.reduce_mean(cost_i)

    return z3, cost


def multi_xavier_1(x, y, _):
    w1 = tf.get_variable('w1', [784, 256],
            initializer=tf.contrib.layers.xavier_initializer())
    w2 = tf.get_variable('w2', [256, 256],
            initializer=tf.contrib.layers.xavier_initializer())
    w3 = tf.get_variable('w3', [256, 10],
            initializer=tf.contrib.layers.xavier_initializer())

    b1 = tf.Variable(tf.random_normal([256]))
    b2 = tf.Variable(tf.random_normal([256]))
    b3 = tf.Variable(tf.random_normal([10]))

    z1 = tf.matmul(x, w1) + b1
    r1 = tf.nn.relu(z1)

    z2 = tf.matmul(r1, w2) + b2
    r2 = tf.nn.relu(z2)

    z3 = tf.matmul(r2, w3) + b3
    # hx = tf.nn.softmax(z3)
    cost_i = tf.nn.softmax_cross_entropy_with_logits(logits=z3,
                                                     labels=y)
    cost = tf.reduce_mean(cost_i)

    return z3, cost


def multi_xavier_2(x, y, _):
    w1 = tf.get_variable('w1_', [784, 256],
            initializer=tf.contrib.layers.xavier_initializer())
    w2 = tf.get_variable('w2_', [256, 256],
            initializer=tf.contrib.layers.xavier_initializer())
    w3 = tf.get_variable('w3_', [256, 10],
            initializer=tf.contrib.layers.xavier_initializer())

    b1 = tf.Variable(tf.zeros([256]))
    b2 = tf.Variable(tf.zeros([256]))
    b3 = tf.Variable(tf.zeros([10]))

    z1 = tf.matmul(x, w1) + b1
    r1 = tf.nn.relu(z1)

    z2 = tf.matmul(r1, w2) + b2
    r2 = tf.nn.relu(z2)

    z3 = tf.matmul(r2, w3) + b3
    # hx = tf.nn.softmax(z3)
    cost_i = tf.nn.softmax_cross_entropy_with_logits(logits=z3,
                                                     labels=y)
    cost = tf.reduce_mean(cost_i)

    return z3, cost


def multi_dropout(x, y, keep_rate):
    w1 = tf.get_variable('w1__', [784, 256],
            initializer=tf.contrib.layers.xavier_initializer())
    w2 = tf.get_variable('w2__', [256, 256],
            initializer=tf.contrib.layers.xavier_initializer())
    w3 = tf.get_variable('w3__', [256, 256],
            initializer=tf.contrib.layers.xavier_initializer())
    w4 = tf.get_variable('w4__', [256, 256],
            initializer=tf.contrib.layers.xavier_initializer())
    w5 = tf.get_variable('w5__', [256, 10],
            initializer=tf.contrib.layers.xavier_initializer())

    b1 = tf.Variable(tf.zeros([256]))
    b2 = tf.Variable(tf.zeros([256]))
    b3 = tf.Variable(tf.zeros([256]))
    b4 = tf.Variable(tf.zeros([256]))
    b5 = tf.Variable(tf.zeros([10]))

    z1 = tf.matmul(x, w1) + b1
    r1 = tf.nn.relu(z1)
    d1 = tf.nn.dropout(r1, keep_rate)

    z2 = tf.matmul(d1, w2) + b2
    r2 = tf.nn.relu(z2)
    d2 = tf.nn.dropout(r2, keep_rate)

    z3 = tf.matmul(d2, w3) + b3
    r3 = tf.nn.relu(z3)
    d3 = tf.nn.dropout(r3, keep_rate)

    z4 = tf.matmul(d3, w4) + b4
    r4 = tf.nn.relu(z4)
    d4 = tf.nn.dropout(r4, keep_rate)

    z5 = tf.matmul(d4, w5) + b5
    # hx = tf.nn.softmax(z5)
    cost_i = tf.nn.softmax_cross_entropy_with_logits(logits=z5,
                                                     labels=y)
    cost = tf.reduce_mean(cost_i)

    return z5, cost


def multi_loops(x, y, keep_rate):
    layers = [784, 256, 256, 10]
    last = len(layers) - 1

    for i in range(last):
        w = tf.get_variable('w__' + str(i),
                            [layers[i], layers[i+1]],
                initializer=tf.contrib.layers.xavier_initializer())
        b = tf.Variable(tf.zeros(layers[i+1]))

        z = tf.matmul(x, w) + b
        if i == last-1:
            break

        r = tf.nn.relu(z)
        x = tf.nn.dropout(r, keep_rate)

    # hx = tf.nn.softmax(z)
    cost_i = tf.nn.softmax_cross_entropy_with_logits(logits=z,
                                                     labels=y)
    cost = tf.reduce_mean(cost_i)

    return z, cost


def show_model(model):
    mnist = input_data.read_data_sets('mnist', one_hot=True)

    x = tf.placeholder(tf.float32)
    y = tf.placeholder(tf.float32)
    keep_rate = tf.placeholder(tf.float32)      # dropout

    # ---------------------------------- #

    hx, cost = model(x, y, keep_rate)

    # ---------------------------------- #

    # optimizer = tf.train.GradientDescentOptimizer(0.01)
    # optimizer = tf.train.GradientDescentOptimizer(0.1)
    optimizer = tf.train.AdamOptimizer(0.001)
    train = optimizer.minimize(cost)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    epochs, batch_size = 15, 100
    iter = mnist.train.num_examples // batch_size

    for i in range(epochs):
        total = 0
        for j in range(iter):
            xx, yy = mnist.train.next_batch(batch_size)

            feed = {x: xx, y: yy, keep_rate: 0.7}
            _, loss = sess.run([train, cost], feed)

            total += loss

        print('{:2} : {}'.format(i, total / iter))

    print('-' * 30)

    show_accuracy(hx, sess, x, y, keep_rate, 'train', mnist.train)
    show_accuracy(hx, sess, x, y, keep_rate, 'valid', mnist.validation)
    show_accuracy(hx, sess, x, y, keep_rate, 'test ', mnist.test)


# show_model(softmax)
# show_model(multi_relu_1)
# show_model(multi_relu_2)
# show_model(multi_xavier_1)
# show_model(multi_xavier_2)
# show_model(multi_dropout)
show_model(multi_loops)




print('\n\n\n\n\n\n\n')



# [1] softmax  0.01
# train : 0.9016182
# valid : 0.9086
# test  : 0.908

# [1] softmax  0.1
# train : 0.92434543
# valid : 0.9264
# test  : 0.9238

# [1] softmax  0.001 adam
# train : 0.9321091
# valid : 0.9296
# test  : 0.9274

# [2] multi_relu_1 256
# train : 0.9748
# valid : 0.9306
# test  : 0.9234

# [2] multi_relu_1 512 성능 저하
# train : 0.95705456
# valid : 0.9044
# test  : 0.907

# [3] multi_relu_2
# train : 0.97194546
# valid : 0.9268
# test  : 0.9237

# [4] multi_xavier_1
# train : 0.9422182
# valid : 0.9454
# test  : 0.9412

# [5] multi_xavier_2
# train : 0.95272726
# valid : 0.955
# test  : 0.9498

# [6] multi_dropout  0.01
# train : 0.95965457
# valid : 0.9602
# test  : 0.9583

# [6] multi_dropout  0.1
# train : 0.9938
# valid : 0.981
# test  : 0.9809

# [6] multi_dropout  0.001  adam
# train : 0.9959091
# valid : 0.984
# test  : 0.982

# [7] multi_loops  3-layers  dropout  adam
# train : 0.9971273
# valid : 0.9834
# test  : 0.9827
