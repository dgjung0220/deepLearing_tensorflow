# Day_04_08_scope.py
import tensorflow as tf

with tf.variable_scope('foo'):
    with tf.variable_scope('bar'):
        v = tf.get_variable('v', shape=[1])

        print(v)    # <tf.Variable 'foo/bar/v:0' shape=(1,) dtype=float32_ref>
        assert v.name == 'foo/bar/v:0'


def make_variable():
    w = tf.get_variable('w', shape=[1])
    # tf.Variable(3, name='k')
    return w


# make_variable()
# make_variable()

with tf.variable_scope('v_1'):
    print('[1]', make_variable())
with tf.variable_scope('v_2'):
    print('[2]', make_variable())
with tf.variable_scope('v_1', reuse=True):
    print('[3]', make_variable())
with tf.variable_scope('v_1') as scope:
    scope.reuse_variables()
    print('[4]', make_variable())
with tf.variable_scope(scope, reuse=True):
    print('[5]', make_variable())

print('[6]', make_variable())
