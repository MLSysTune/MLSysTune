import time
import logging
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from absl import app
from selftf.lib.mltuner.mltuner_util import MLTunerUtil
from selftf.lib.mltuner.mltuner_util import convert_model



logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M',
                    )

flags = tf.compat.v1.app.flags
# end defination of input argument

mltunerUtil = MLTunerUtil()
FLAGS = flags.FLAGS


def main(argv):
    batch_size = mltunerUtil.get_batch_size()
    learning_rate = mltunerUtil.get_learning_rate()

    df_newsgroup = pd.read_csv('/root/cyliu/tftuner/selftf/tf_job/nlp/zmwu/bd_lstm/20newsgroup_preprocessed.csv', sep=';', usecols=['target', 'text_cleaned'])
    df_newsgroup.rename(columns={'text_cleaned' : 'text'}, inplace=True)

    le = LabelEncoder()
    le.fit(df_newsgroup['target'].unique())
    df_newsgroup['target'] = le.transform(df_newsgroup['target'])

    X = df_newsgroup['text'].astype(str)
    y = tf.keras.utils.to_categorical(df_newsgroup['target'], num_classes=df_newsgroup['target'].nunique())

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=df_newsgroup['target'])

    tokenizer = tf.keras.preprocessing.text.Tokenizer()
    tokenizer.fit_on_texts(X_train)

    vocab_size = len(tokenizer.word_index) + 1

    train_seq = tokenizer.texts_to_sequences(X_train)
    test_seq = tokenizer.texts_to_sequences(X_test)


    max_length = len(max(train_seq, key=len))

    train_vector = tf.keras.preprocessing.sequence.pad_sequences(train_seq, maxlen=max_length, padding='post', truncating='post')
    test_vector = tf.keras.preprocessing.sequence.pad_sequences(test_seq, maxlen=max_length, padding='post', truncating='post')


    model = tf.keras.models.Sequential([
        tf.keras.layers.Embedding(vocab_size, 64, input_length=max_length),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(20, activation='softmax')
    ])
        
    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    if FLAGS.get_model:
        convert_model(model,FLAGS.script_path)
        return

    model_dir = "{}/{}".format("/opt/tftuner", mltunerUtil.get_job_id())
    strategy = tf.distribute.experimental.ParameterServerStrategy()
    session_config = mltunerUtil.get_tf_session_config()
    config = tf.estimator.RunConfig(train_distribute=strategy,
                                    session_config=session_config,
                                    save_summary_steps=None,
                                    save_checkpoints_secs=None,
                                    model_dir=model_dir)
    keras_estimator = tf.keras.estimator.model_to_estimator(keras_model=model,config=config)


    # Begin input
    def train_input_fn():
        dataset = tf.data.Dataset.from_tensor_slices((train_vector, y_train))
        dataset = dataset.shard(mltunerUtil.get_num_worker(),mltunerUtil.get_worker_index())
        dataset = dataset.repeat()
        dataset = dataset.shuffle(batch_size * mltunerUtil.get_num_worker())
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(mltunerUtil.get_num_worker())
        return dataset

    def eval_input_fn():
        dataset = tf.data.Dataset.from_tensor_slices((test_vector, y_test))
        dataset = dataset.shard(mltunerUtil.get_num_worker(),mltunerUtil.get_worker_index())
        dataset = dataset.repeat()
        dataset = dataset.shuffle(batch_size * mltunerUtil.get_num_worker())
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(mltunerUtil.get_num_worker())
        return dataset

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

    train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=10,hooks=[LoggerHook()])
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
