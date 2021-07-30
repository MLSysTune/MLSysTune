import tensorflow as tf


c = tf.constant("Hello, distributed TensorFlow!")
v = tf.get_variable("a", [1], dtype=tf.int32, initializer=tf.zeros_initializer)
const = tf.get_variable("b", [1], dtype=tf.int32, initializer=tf.ones_initializer)
a = tf.assign(v, tf.add(v, const))

server = tf.train.Server.create_local_server()
sess = tf.Session(server.target)  # Create a session on the server.
sess.run(tf.global_variables_initializer())

sess.run(a)
print(sess.run(v))

sess.run(a)
print(sess.run(v))

sess.run(a)
print(sess.run(v))