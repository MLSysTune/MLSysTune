import tensorflow as tf

def split_tf_record(src, dst, shards):
    raw_dataset = tf.data.TFRecordDataset(src)
    sess = tf.Session()
    for i in range(shards):
        writer = tf.data.experimental.TFRecordWriter("{}/cifar10-train.tfrecord-00000-of-00001".format(dst, i))
        sess.run(writer.write(raw_dataset.shard(shards, i)))

dst = "/tmp/cifar_tfrecord"
split_tf_record("/Users/cyliu/tensorflow_datasets/cifar10/3.0.2/cifar10-train.tfrecord-00000-of-00001",
                dst,
                500)