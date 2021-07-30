import tensorflow as tf
import tensorflow.contrib.graph_editor as tfge
import re

a = tf.Variable(1, name="a")
# a = tf.get_variable("a", initializer=tf.zeros(4), partitioner=tf.fixed_size_partitioner(2))
b = tf.Variable(2, name="b")


plus = tf.add(a,a, "plus")
graph = tf.get_default_graph()
with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  ret = sess.run(plus)
  print(ret)
  assert ret == 2

tensors = tfge.get_tensors(graph)

def get_name_scope_from_variable(v):

  regex = re.compile("^(.*):.*$")
  m = regex.match(v.name)
  return m.group(1)

a_ref = a.name
b_ref = b.name
# a_scope = get_name_scope_from_variable(a)
# assert(a_scope == "a")
read_a = tfge.select_ts("a/read", graph=tf.get_default_graph())
read_b = tfge.select_ts("b/read", graph=tf.get_default_graph())


# tfge.swap_ts(read_a, read_b)
# tfge.swap_outputs(read_a, read_b)
# tfge.reroute_outputs(read_b, read_a)
tfge.reroute_inputs(read_b, read_a)


summary_writer = tf.summary.FileWriter("logdir/", graph=tf.get_default_graph())

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  ret = sess.run(plus)
  print(ret)
  assert ret == 4
