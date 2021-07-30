import tensorflow as tf
from selftf.lib.ml_job_framework import MLJobFramework
import selftf.tf_job.inception.inception_model as inception
from selftf.tf_job.inception.slim import slim
import selftf.tf_job.inception.image_processing as image_processing
from selftf.tf_job.inception.imagenet_data import ImagenetData

num_preprocess_threads = 4
dataset = ImagenetData(subset="train")


class InceptionV3(MLJobFramework):

    def model_definition(self, context):
        # Variables and its related init/assign ops are assigned to ps.
        # with slim.scopes.arg_scope(
        #     [slim.variables.variable, slim.variables.global_step],
        #     device=slim.variables.VariableDeviceChooser(num_parameter_servers)):
        # Create a variable to count the number of train() calls. This equals the
        # number of updates applied to the variables.
        global_step = slim.variables.global_step()
        context.set_global_step(global_step)

        # Calculate the learning rate schedule.
        # num_batches_per_epoch = (dataset.num_examples_per_epoch() /
        #                          FLAGS.batch_size)
        # Decay steps need to be divided by the number of replicas to aggregate.
        # decay_steps = int(num_batches_per_epoch * FLAGS.num_epochs_per_decay /
        #                   num_replicas_to_aggregate)

        # Decay the learning rate exponentially based on the number of steps.
        # lr = tf.train.exponential_decay(FLAGS.initial_learning_rate,
        #                                 global_step,
        #                                 decay_steps,
        #                                 FLAGS.learning_rate_decay_factor,
        #                                 staircase=True)
        # Add a summary to track the learning rate.
        # tf.summary.scalar('learning_rate', lr)

        # # Create an optimizer that performs gradient descent.
        # opt = tf.train.RMSPropOptimizer(lr,
        #                                 RMSPROP_DECAY,`
        #                                 momentum=RMSPROP_MOMENTUM,
        #                                 epsilon=RMSPROP_EPSILON)
        images, labels = image_processing.distorted_inputs(
            dataset,
            batch_size=context.get_batch_size(),
            num_preprocess_threads=num_preprocess_threads)

        # Number of classes in the Dataset label set plus 1.
        # Label 0 is reserved for an (unused) background class.
        num_classes = dataset.num_classes() + 1
        logits = inception.inference(images, num_classes, for_training=True)
        # Add classification loss.
        inception.loss(logits, labels, batch_size=context.get_batch_size())

        # Gather all of the losses including regularization losses.
        losses = tf.get_collection(slim.losses.LOSSES_COLLECTION)
        losses += tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)

        total_loss = tf.add_n(losses, name='total_loss')

        if context.get_is_chief():
            # Compute the moving average of all individual losses and the
            # total loss.
            loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
            loss_averages_op = loss_averages.apply(losses + [total_loss])

            # Add dependency to compute loss_averages.
            with tf.control_dependencies([loss_averages_op]):
                total_loss = tf.identity(total_loss)

        exp_moving_averager = tf.train.ExponentialMovingAverage(
            inception.MOVING_AVERAGE_DECAY, global_step)

        variables_to_average = (
                tf.trainable_variables() + tf.moving_average_variables())

        batchnorm_updates = tf.get_collection(slim.ops.UPDATE_OPS_COLLECTION)
        assert batchnorm_updates, 'Batchnorm updates are missing'
        batchnorm_updates_op = tf.group(*batchnorm_updates)
        # Add dependency to compute batchnorm_updates.
        with tf.control_dependencies([batchnorm_updates_op]):
            total_loss = tf.identity(total_loss)

        # # Compute gradients with respect to the loss.
        # grads = opt.compute_gradients(total_loss)
        #
        # # Add histograms for gradients.
        # for grad, var in grads:
        #   if grad is not None:
        #     tf.summary.histogram(var.op.name + '/gradients', grad)
        #
        # apply_gradients_op = opt.apply_gradients(grads, global_step=global_step)
        #
        # with tf.control_dependencies([apply_gradients_op]):
        #   train_op = tf.identity(total_loss, name='train_op')

        # Get chief queue_runners and init_tokens, which is used to synchronize
        # replicas. More details can be found in SyncReplicasOptimizer.

        context.set_train_op(total_loss,
                             sync=True,
                             sync_exp_moving_averager=exp_moving_averager,
                             variables_to_average=variables_to_average)


if __name__ == "__main__":
    model = InceptionV3()
    model.run()
