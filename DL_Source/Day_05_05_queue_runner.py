# Day_05_05_queue_runner.py
import tensorflow as tf


def basic():
    # queue = tf.train.string_input_producer(['12', '34', '56'])
    queue = tf.train.string_input_producer(['12', '34', '56'],
                                           shuffle=False)

    sess = tf.Session()
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess, coord)

    for i in range(20):
        value = sess.run(queue.dequeue())
        print(value, value.decode('utf-8'))

        if i % 3 == 2:
            print()

    coord.request_stop()
    coord.join(threads)


def advanced():
    queue = tf.train.string_input_producer(['Data/q1.txt',
                                            'Data/q2.txt',
                                            'Data/q3.txt'],
                                           shuffle=False)

    sess = tf.Session()
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess, coord)

    reader = tf.TextLineReader()
    key, value = reader.read(queue)
    print('key   :', sess.run(key))
    print('value :', sess.run(value))

    record_defaults = [[0.], [0.], [99.]]
    for i in range(20):
        x1, x2, x3 = tf.decode_csv(value,
                                   record_defaults)
        print(sess.run([x1, x2, x3]))

    coord.request_stop()
    coord.join(threads)


def iris_softmax():
    queue = tf.train.string_input_producer(['Data/iris_softmax.csv'])

    reader = tf.TextLineReader()
    _, value = reader.read(queue)

    record_defaults = [[0.]] * 8
    iris = tf.decode_csv(value, record_defaults)

    batch_size = 5
    x_batch, y_batch = tf.train.batch([iris[:-3], iris[-3:]],
                                      batch_size)

    # -------------------------------------- #

    x = tf.placeholder(tf.float32)
    y = tf.placeholder(tf.float32)

    w = tf.Variable(tf.zeros([5, 3]))
    b = tf.Variable(tf.zeros([3]))

    z = tf.matmul(x, w) + b
    hx = tf.nn.softmax(z)

    cost_i = tf.nn.softmax_cross_entropy_with_logits(logits=z,
                                                     labels=y)
    cost = tf.reduce_mean(cost_i)

    optimizer = tf.train.AdamOptimizer(0.001)
    train = optimizer.minimize(cost)

    sess = tf.Session()
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess, coord)

    sess.run(tf.global_variables_initializer())

    epochs = 15
    iter = 150 // batch_size

    for i in range(epochs):
        total = 0
        for j in range(iter):
            xx, yy = sess.run([x_batch, y_batch])

            feed = {x: xx, y: yy}
            _, loss = sess.run([train, cost], feed)

            total += loss

        print('{:2} : {}'.format(i, total / iter))

    coord.request_stop()
    coord.join(threads)
    sess.close()

# basic()
# advanced()
iris_softmax()
