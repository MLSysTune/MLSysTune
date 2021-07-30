import logging
import tensorflow as tf
from tensorflow import keras
import numpy as np
import time
from absl import flags, app
from selftf.lib.mltuner.mltuner_util import MLTunerUtil

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M',
                    )


mltunerUtil = MLTunerUtil()

def main(argv):

    vocab_size = 10000
    batch_size = mltunerUtil.get_batch_size()
    learning_rate = mltunerUtil.get_learning_rate()

    # Load dataset
    (x_train_variable, y_train), (x_test_variable, y_test) = keras.datasets.imdb.load_data(path="/root/cyliu/tftuner/selftf/tf_job/nlp/zmwu/imdb/data/imdb.npz",num_words=vocab_size)

    # a dict map word to integer index
    word_index = keras.datasets.imdb.get_word_index(path="/root/cyliu/tftuner/selftf/tf_job/nlp/zmwu/imdb/data/imdb_word_index.json")

    # keep the first index
    word_index = {k:(v+3) for k,v in word_index.items()}
    word_index["<PAD>"] = 0
    word_index["<START>"] = 1
    word_index["<UNK>"] = 2  # unknown
    word_index["<UNUSED>"] = 3

    train_data = keras.preprocessing.sequence.pad_sequences(x_train_variable,
                                                            value=word_index["<PAD>"],
                                                            padding='post',
                                                            maxlen=256)
    test_data = keras.preprocessing.sequence.pad_sequences(x_test_variable,
                                                           value=word_index["<PAD>"],
                                                           padding='post',
                                                           maxlen=256)

    # Begin input
    def train_input_fn():
        dataset = tf.data.Dataset.from_tensor_slices((train_data, y_train))
        dataset = dataset.shard(mltunerUtil.get_num_worker(),mltunerUtil.get_worker_index())
        dataset = dataset.repeat()
        dataset = dataset.shuffle(batch_size * mltunerUtil.get_num_worker())
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(mltunerUtil.get_num_worker())
        return dataset

    def eval_input_fn():
        dataset = tf.data.Dataset.from_tensor_slices((test_data, y_test))
        dataset = dataset.shard(mltunerUtil.get_num_worker(),mltunerUtil.get_worker_index())
        dataset = dataset.repeat()
        dataset = dataset.shuffle(batch_size * mltunerUtil.get_num_worker())
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(mltunerUtil.get_num_worker())
        return dataset

    # End input


    # Begin model def
    model = keras.Sequential()
    model.add(keras.layers.Embedding(vocab_size, 16))
    model.add(keras.layers.GlobalAveragePooling1D())
    model.add(keras.layers.Dense(16, activation='relu'))
    model.add(keras.layers.Dense(1, activation='sigmoid'))
    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=opt,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    model.summary()

    # End model def 


    model_dir = "{}/{}".format("/opt/tftuner", mltunerUtil.get_job_id())

    strategy = tf.distribute.experimental.ParameterServerStrategy()
    session_config = mltunerUtil.get_tf_session_config()
    config = tf.estimator.RunConfig(train_distribute=strategy,
                                    session_config=session_config,
                                    save_summary_steps=None,
                                    save_checkpoints_secs=None,
                                    model_dir=model_dir)
    # config = tf.estimator.RunConfig(save_summary_steps=None,save_checkpoints_steps=100,model_dir=model_dir)

    # Begin convert keras to estimator
    keras_estimator = tf.keras.estimator.model_to_estimator(keras_model=model,config=config)
    # End convert keras to estimator

    class LoggerHook(tf.estimator.SessionRunHook):
        """Logs loss and runtime."""

        def __init__(self):
            self.last_run_timestamp = time.time()
        
        def after_run(self, run_context, run_values):
            session: tf.Session = run_context.session
            loss, step = session.run([tf.compat.v1.get_collection("losses")[0],
                                      tf.compat.v1.get_collection("global_step_read_op_cache")[0]])
            logging.debug("step:{} loss:{}".format(step, loss))
            mltunerUtil.report_iter_loss(step, loss,
                                         time.time() - self.last_run_timestamp)
            self.last_run_timestamp = time.time()


    # Begin training
    train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=1000000,hooks=[LoggerHook()])
    eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn)

    # wait for chief ready?
    if not (mltunerUtil.is_chief() or mltunerUtil.is_ps()):
        time.sleep(1)
        if not tf.io.gfile.exists(model_dir):
            logging.debug("wait for chief init")
            time.sleep(1)


    tf.estimator.train_and_evaluate(keras_estimator,train_spec,eval_spec)


if __name__ == '__main__':
  app.run(main)

