import tensorflow as tf


def weight(shape, name):
    w = tf.get_variable(name=name, shape=shape,
                        initializer=tf.truncated_normal_initializer(stddev=0.01, dtype=tf.float32))
    tf.add_to_collection('weights', w)
    return w


def bias(shape, value, name):
    b = tf.get_variable(name=name, shape=shape,
                        initializer=tf.constant_initializer(value))
    return b


def inference(x, keep_prob=0.5):

    # tf.Assert(x.shape == [None, 224, 224, 3], [x])

    with tf.variable_scope('layer1') as scope:
        # group1
        kernel_conv1_g1 = weight(shape=[11, 11, 3, 48], name='kernel_conv1_g1')
        bias_conv1_g1 = bias(shape=[48], value=0, name='bias_conv1_g1')
        h_conv1_g1 = tf.nn.relu(tf.nn.conv2d(x, kernel_conv1_g1, strides=[1, 4, 4, 1],
                                padding='SAME') + bias_conv1_g1)
        h_lrn1_g1 = tf.nn.local_response_normalization(h_conv1_g1, depth_radius=5,
                                                    bias=2, alpha=1e-04, beta=0.75)
        h_pool1_g1 = tf.nn.max_pool(h_lrn1_g1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                                    padding='VALID')
        # tf.Assert(h_pool1_g1.shape == [None, 27, 27, 48], [h_pool1_g1])
        # group 2
        kernel_conv1_g2 = weight(shape=[11, 11, 3, 48], name='kernel_conv1_g2')
        bias_conv1_g2 = bias(shape=[48], value=0, name='bias_conv1_g2')
        h_conv1_g2 = tf.nn.relu(tf.nn.conv2d(x, kernel_conv1_g2, strides=[1, 4, 4, 1],
                                             padding='SAME') + bias_conv1_g2)
        h_lrn1_g2 = tf.nn.local_response_normalization(h_conv1_g2, depth_radius=5,
                                                         bias=2, alpha=1e-04, beta=0.75)
        h_pool1_g2 = tf.nn.max_pool(h_lrn1_g2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                                    padding='VALID')
        # tf.Assert(h_pool1_g2.shape == [None, 27, 27, 48], [h_pool1_g2])

    with tf.variable_scope('layer2') as scope:
        # group 1
        kernel_conv2_g1 = weight(shape=[5, 5, 48, 128], name='kernel_conv2_g1')
        bias_conv2_g1 = bias(shape=[128], value=1, name='bias_conv2_g1')
        h_conv2_g1 = tf.nn.relu(tf.nn.conv2d(h_pool1_g1, kernel_conv2_g1, strides=[1, 1, 1, 1],
                                padding='SAME') + bias_conv2_g1)
        h_lrn2_g1 = tf.nn.local_response_normalization(h_conv2_g1, depth_radius=5,
                                                       bias=2, alpha=1e-04, beta=0.75)
        h_pool2_g1 = tf.nn.max_pool(h_lrn2_g1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                                    padding='VALID')
        # tf.Assert(h_pool2_g1 == [None, 13, 13, 128], [h_pool2_g1])
        # group 2
        kernel_conv2_g2 = weight(shape=[5, 5, 48, 128], name='kernel_conv2_g2')
        bias_conv2_g2 = bias(shape=[128], value=1, name='bias_conv2_g2')
        h_conv2_g2 = tf.nn.relu(tf.nn.conv2d(h_pool1_g2, kernel_conv2_g2, strides=[1, 1, 1, 1],
                                             padding='SAME') + bias_conv2_g2)
        h_lrn2_g2 = tf.nn.local_response_normalization(h_conv2_g2, depth_radius=5,
                                                       bias=2, alpha=1e-04, beta=0.75)
        h_pool2_g2 = tf.nn.max_pool(h_lrn2_g2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                                    padding='VALID')
        # tf.Assert(h_pool2_g2 == [None, 13, 13, 128], [h_pool2_g2])

    with tf.variable_scope('layer3') as scope:
        h_pool2 = tf.concat([h_pool2_g1, h_pool2_g2], -1)
        # group 1
        kernel_conv3_g1 = weight(shape=[3, 3, 256, 192], name='kernel_conv3_g1')
        bias_conv3_g1 = bias(shape=[192], value=0, name='bias_conv3_g1')
        h_conv3_g1 = tf.nn.relu(tf.nn.conv2d(h_pool2, kernel_conv3_g1, strides=[1, 1, 1, 1],
                                             padding='SAME') + bias_conv3_g1)
        # tf.Assert(h_conv3_g1.shape == [None, 13, 13, 192], [h_conv3_g1])
        # group 2
        kernel_conv3_g2 = weight(shape=[3, 3, 256, 192], name='kernel_conv3_g2')
        bias_conv3_g2 = bias(shape=[192], value=0, name='bias_conv3_g2')
        h_conv3_g2 = tf.nn.relu(tf.nn.conv2d(h_pool2, kernel_conv3_g2, strides=[1, 1, 1, 1], padding='SAME')
                                + bias_conv3_g2)
        # tf.Assert(h_conv3_g2.shape == [None, 13, 13, 192], [h_conv3_g2])

    with tf.variable_scope('layer4') as scope:
        # group 1
        kernel_conv4_g1 = weight(shape=[3, 3, 192, 192], name='kernel_conv4_g1')
        bias_conv4_g1 = bias(shape=[192], value=1, name='bias_conv4_g1')
        h_conv4_g1 = tf.nn.relu(tf.nn.conv2d(h_conv3_g1, kernel_conv4_g1, strides=[1, 1, 1, 1],
                                             padding='SAME') + bias_conv4_g1)
        # tf.Assert(h_conv4_g1.shape == [None, 13, 13, 192], [h_conv4_g1])
        # group 2
        kernel_conv4_g2 = weight(shape=[3, 3, 192, 192], name='kernel_conv4_g2')
        bias_conv4_g2 = bias(shape=[192], value=1, name='bias_conv4_g2')
        h_conv4_g2 = tf.nn.relu(tf.nn.conv2d(h_conv3_g2, kernel_conv4_g2, strides=[1, 1, 1, 1],
                                             padding='SAME') + bias_conv4_g2)
        tf.Assert(False, [h_conv4_g2])
        # tf.Assert(h_conv4_g2.shape == [None, 13, 13, 192], [h_conv4_g2])

    with tf.variable_scope('layer5') as scope:
        # group 1
        kernel_conv5_g1 = weight(shape=[3, 3, 192, 128], name='kernel_conv5_g1')
        bias_conv5_g1 = bias(shape=[128], value=1, name='bias_conv5_g1')
        h_conv5_g1 = tf.nn.relu(tf.nn.conv2d(h_conv4_g1, kernel_conv5_g1, strides=[1, 1, 1, 1],
                                padding='SAME') + bias_conv5_g1)
        h_pool5_g1 = tf.nn.max_pool(h_conv5_g1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                                    padding='VALID')
        # tf.Assert(h_pool5_g1.shape == [None, 6, 6, 128], [h_pool5_g1])
        # group 2
        kernel_conv5_g2 = weight(shape=[3, 3, 192, 128], name='kernel_conv5_g2')
        bias_conv5_g2 = bias(shape=[128], value=1, name='bias_conv5_g2')
        h_conv5_g2 = tf.nn.relu(tf.nn.conv2d(h_conv4_g2, kernel_conv5_g2, strides=[1, 1, 1, 1],
                                             padding='SAME') + bias_conv5_g2)
        h_pool5_g2 = tf.nn.max_pool(h_conv5_g2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                                    padding='VALID')
        # tf.Assert(h_pool5_g2.shape == [None, 6, 6, 128], [h_pool5_g2])

    with tf.variable_scope('layer6') as scope:
        h_pool5 = tf.concat([h_pool5_g1, h_pool5_g2], -1)
        h_pool5_flat = tf.reshape(h_pool5, [-1, 6*6*256])
        w_fc1 = weight(shape=[6*6*256, 4096], name='w_fc1')
        b_fc1 = bias(shape=[4096], value=1, name='b_fc1')
        h_fc1 = tf.nn.relu(tf.matmul(h_pool5_flat, w_fc1) + b_fc1)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    with tf.variable_scope('layer7') as scope:
        w_fc2 = weight(shape=[4096, 4096], name='w_fc2')
        b_fc2 = bias(shape=[4096], value=1, name='b_fc2')
        h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, w_fc2) + b_fc2)
        h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)

    with tf.variable_scope('output_layer') as scope:
        w_fc3 = weight(shape=[4096, 1000], name='w_fc3')
        b_fc3 = bias(shape=[1000], value=0, name='b_fc3')
        logits = tf.matmul(h_fc2_drop, w_fc3) + b_fc3
        softmax = tf.nn.softmax(logits)

    return logits, softmax




