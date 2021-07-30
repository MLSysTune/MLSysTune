import time

import tensorflow as tf
import tensorflow_datasets as tfds

sess = tf.Session()
split = tfds.Split.TRAIN

dataset,info = tfds.load('cifar10', split=split, as_supervised=True,with_info=True)
info: tfds.core.DatasetInfo = info
split_info:tfds.core.SplitInfo= info.splits[str(split)]
image_shape = info.features["image"].shape
label_info = info.features["label"]
num_class = info.features["label"].num_classes
image_shape = info.features["image"].shape


num_examples = split_info.num_examples

dataset = dataset.shard(5000,10).cache().repeat()
iter = tf.data.make_one_shot_iterator(dataset)
counter = 0
op = iter.get_next()
start_timestamp = time.time()
try:
    while(True):
        data_tensor = sess.run(op)
        counter += 1
        print(f"{counter}")
except:
    print(f"Total time: {time.time()-start_timestamp}")
