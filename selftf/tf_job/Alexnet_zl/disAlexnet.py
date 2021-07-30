from selftf.lib.ml_job_framework import MLJobFramework
from selftf.tf_job.Alexnet_zl.Alexnet_structure_paper import inference
import tensorflow as tf
from selftf.tf_job.cnn.AlexNet import train_util as tu


#TODO: Data Augmentation

class Alexnet_imagenet(MLJobFramework):

    def __init__(self):
        MLJobFramework.__init__(self)
        self.x = None
        self.y_ = None
        self.wnid_labels = None
        self.accuracy = None

    def model_definition(self, context):

        # keep_prob = context.get_keep_prob() #no corresponding method

        global_step = tf.get_variable('global_step', [], 
                                      initializer=tf.constant_initializer(1),
                                      trainable=False)
        self.wnid_labels = tu.load_imagenet_meta('hdfs://ssd02:8020/user/root/train_data/imagenet/meta_data.txt')

        self.x = tf.placeholder(tf.float32, shape=[None, 224, 224, 3])
        self.y_ = tf.placeholder(tf.float32, shape=[None, 1000])

        y_pred, _ = inference(self.x)
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y_, logits=y_pred))
        l2_loss = tf.reduce_sum(5e-04 * tf.stack([tf.nn.l2_loss(v) for v in tf.get_collection('weights')]))
        loss = cross_entropy + l2_loss
        
        with tf.variable_scope('train'):
            context.set_train_op(loss=loss, global_step=global_step)

        with tf.variable_scope('Accuracy'):
            correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(self.y_, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def get_feed_dict(self, context):
        batch_size = context.get_batch_size()
        val_x, val_y = tu.read_batch(batch_size,
                                     "hdfs://ssd02:8020/user/root/train_data/imagenet/train/",
                                     self.wnid_labels)
        return {
                self.x: val_x,
                self.y_: val_y,
               }


model = Alexnet_imagenet()
model.run()


