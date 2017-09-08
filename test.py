import tensorflow as tf

var = tf.Variable(0, dtype=tf.int32)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    var = tf.assign(var, 10)
    print sess.run(var)