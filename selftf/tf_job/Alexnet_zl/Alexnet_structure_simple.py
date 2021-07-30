import tensorflow as tf


def weight(shape, name):
    w = tf.get_variable(name=name, shape=shape,
        initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
    return w

def bias(shape, name):
    b = tf.get_variable(name=name, shape=shape,
        initializer=tf.constant_initializer(0))
    return b

def inference(x, keep_prob=0.5):
    with tf.variable_scope('layer1') as scope:
        kernel_conv1 = weight(shape=[11,11,3,96], name='kernel_conv1')
        bias_conv1 = bias(shape=[96], name='bias_conv1')
        h_conv1 = tf.nn.relu(tf.nn.conv2d(x, kernel_conv1, strides=[1,4,4,1], 
                             padding='VALID') + bias_conv1)
        h_lrn1 = tf.nn.local_response_normalization(h_conv1, depth_radius=5,
                                                    bias=2, alpha=1e-04, beta=0.75)
        h_pool1 = tf.nn.max_pool(h_lrn1, ksize=[1,3,3,1], strides=[1,2,2,1],
                                 padding='VALID')

    with tf.variable_scope('layer2') as scope:
        # this layer does not comply the description in the paper
        kernel_conv2 = weight(shape=[5,5,96,256], name='kernel_conv2')
        bias_conv2 = bias(shape=[256], name='bias_conv2')
        h_conv2 = tf.nn.relu(tf.nn.conv2d(h_pool1, kernel_conv2, strides=[1,1,1,1],
                             padding='SAME') + bias_conv2)
        h_lrn2 = tf.nn.local_response_normalization(h_conv2, depth_radius=5,
                                                     bias=2, alpha=1e-04, beta=0.75)
        h_pool2 = tf.nn.max_pool(h_lrn2, ksize=[1,3,3,1], strides=[1,2,2,1],
                                 padding='VALID')


    with tf.variable_scope('layer3') as scope:
        kernel_conv3 = weight(shape=[3,3,256,384], name='kernel_conv3')
        bias_conv3 = bias(shape=[384], name='bias_conv3')
        h_conv3 = tf.nn.relu(tf.nn.conv2d(h_pool2, kernel_conv3, strides=[1,1,1,1],
                             padding='SAME') + bias_conv3)

     with tf.variable_scope('layer4') as scope:
        #note: this layer has the same problem as layer2
        kernel_conv4 = weight(shape=[3,3,384,384], name='kernel_conv4')
        bias_conv4 = bias(shape=[384], name='bias_conv4')
        h_conv4 = tf.nn.relu(tf.nn.conv2d(h_conv3, kernel_conv4, strides=[1,1,1,1],
                              padding='SAME') + bias_conv4)

    with tf.variable_scope('layer5') as scope:
        #note: layer5 has the same problem as layer2
        kernel_conv5 = weight(shape=[3,3,384,256], name='kernel_conv5')
        bias_conv5 = bias(shape=[256], name='bias_conv5')
        h_conv5 = tf.nn.relu(tf.nn.conv2d(h_conv4, kernel_conv5, strides=[1,1,1,1],
                            padding='SAME') + bias_conv5)
        h_pool5 = tf.nn.max_pool(h_conv5, ksize=[1,3,3,1], strides=[1,2,2,1],
                                 padding='VALID')

    with tf.variable_scope('layer6') as scope:
        w_fc1 = weight(shape=[6*6*256, 4096], name='w_fc1')
        b_fc1 = bias(shape=[4096], name='b_fc1')
        h_pool5_flat = tf.reshape(h_pool5, [-1, 6*6*256])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool5_flat, w_fc1) + b_fc1)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    with tf.variable_scope('layer7') as scope:
        w_fc2 = weight(shape=[4096,4096], name='w_fc2')
        b_fc2 = bias(shape=[4096], name='b_fc2')
        h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, w_fc2) + b_fc2)
        h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)

    with tf.variable_scope('output_layer') as scope:
        w_fc3 = weight(shape=[4096,1000], name='w_fc3')
        b_fc3 = bias(shape=[1000], name='b_fc3')
        logits = tf.matmul(h_fc2_drop, w_fc3) + b_fc3
        softmax = tf.nn.softmax(logits)

    return logits, softmax






