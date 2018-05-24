# Day_05_03_tensorboard.py
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


mnist = input_data.read_data_sets('mnist', one_hot=True)

with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, name='x')
    y = tf.placeholder(tf.float32, name='y')

with tf.name_scope('weight'):
    w = tf.Variable(tf.zeros([784, 10]), name='w')
    b = tf.Variable(tf.zeros([10]), name='b')

with tf.name_scope('model'):
    z = tf.matmul(x, w) + b
    hx = tf.nn.softmax(z)

    cost_i = tf.nn.softmax_cross_entropy_with_logits(logits=z,
                                                     labels=y)
    cost = tf.reduce_mean(cost_i)

with tf.name_scope('train'):
    optimizer = tf.train.AdamOptimizer(0.001)
    train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# step 1.
tf.summary.scalar('cost', cost)

# step 2.
merged = tf.summary.merge_all()

# step 3.
writer = tf.summary.FileWriter('board/mnist', sess.graph)

epochs, batch_size = 15, 100
iter = mnist.train.num_examples // batch_size

for i in range(epochs):
    total = 0
    for j in range(iter):
        xx, yy = mnist.train.next_batch(batch_size)

        feed = {x: xx, y: yy}
        _, loss = sess.run([train, cost], feed)

        total += loss

    print('{:2} : {}'.format(i, total / iter))

    # step 4.
    summary = sess.run(merged, {x: xx, y: yy})
    writer.add_summary(summary, i)

# step 5.
# tensorboard --logdir=board/mnist






