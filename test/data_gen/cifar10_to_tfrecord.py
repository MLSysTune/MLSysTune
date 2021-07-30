import tensorflow as tf

import pickle
import os
import tensorflow

path = r'cifar-10-batches-py'


def unpickle(file):
    with open(file, 'rb') as f:
        dict = pickle.load(f, encoding='bytes')
    return dict


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def restore(inputfilename, dis_dir, partition=100):
    number_of_image = 10000
    num_of_image_per_partition = number_of_image / partition
    dict = unpickle(inputfilename)
    # dict: include [b'batch_label', b'labels', b'data', b'filenames']
    # we just choose labels and data. And we choose restore it by int64
    # labels:[10000,1]
    labels = dict[b'labels']
    # images:[10000,3072]
    images = dict[b'data']
    src_file_name = os.path.split(inputfilename)[1]
    writer = tf.python_io.TFRecordWriter("{}/{}_{}.tfrecord".format(dis_dir, src_file_name, 0))
    for i in range(number_of_image):
        # image_raw = tf.convert_to_tensor(images[i])
        example = tf.train.Example(features=tf.train.Features(feature={
            'raw_image': _bytes_feature(pickle.dumps(images[i])),
            'label': _int64_feature(labels[i])
        }))
        writer.write(example.SerializeToString())
        if i % num_of_image_per_partition == 0 and i != 0:
            writer.close()
            writer = tf.python_io.TFRecordWriter(
                "{}/{}_{}.tfrecord".format(dis_dir, src_file_name, i))


filenames = [os.path.join(path, 'data_batch_%d' % i) for i in range(1, 6)]
dst = '/tmp/cifar_tfrecord'
if not os.path.exists(dst):
    os.mkdir(dst)

for filename in filenames:
    restore(filename, '/tmp/cifar_tfrecord')