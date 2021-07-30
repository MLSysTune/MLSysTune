# coding=utf-8
import tensorflow as tf
import numpy as np


def read_batch(lines, is_one_hot):
    batch_size = len(lines)
    label_list = []
    ids = []
    sp_indices = []
    weight_list = []
    for i, line in enumerate(lines):
        label, indices, values = parse_line_for_batch_for_libsvm(line, is_one_hot)
        label_list.append(label)
        ids += indices
        # sp_indices = np.array([[i, index] for index in indices])
        for index in indices:
            sp_indices.append([i, index])
        weight_list += values
    if is_one_hot:
        label_list = np.reshape(label_list, (batch_size, 2))
    else:
        label_list = np.reshape(label_list, (batch_size, 1))
    return label_list, \
           sp_indices, \
           weight_list, \
    # lablelist,id_list,


def parse_line_for_batch_for_libsvm(line, is_one_hot):
    value = line.decode("ascii") .split(" ")
    if is_one_hot:
        if value[0] == "1":
            label = [1, 0]
        else:
            label = [0, 1]
    else:
        label = value[0]
    indices = []
    values = []
    for item in value[1:]:
        [index, value] = item.split(':')
        # if index start with 1, index = int(index)-1
        # else index=int(index)
        index = int(index) - 1
        value = float(value)
        indices.append(index)
        values.append(value)
    return label, indices, values

#
# def parse_line_for_batch_for_libsvm2(line):
#     value = line.value.split(" ")
#     label = []
#     label = value[0]
#     indices = []
#     values = []
#     for item in value[1:]:
#         [index, value] = item.split(':')
#         # if index start with 1, index = int(index)-1
#         # else index=int(index)
#         index = int(index) - 1
#         value = float(value)
#         indices.append(index)
#         values.append(value)
#     return label, indices, values


# label:label fo data
# indices: the list of index of featue
# indices: the value of each features
#
# def main():
#     trainset_files = ["/Volumes/Macintosh HD/Users/cyliu/PycharmProjects/fi"]
#     print (trainset_files)
#     train_filename_queue = tf.train.string_input_producer(trainset_files)
#     train_reader = tf.TextLineReader()
#     key_tensor, line_tensor = train_reader.read(train_filename_queue)
#     train_data_batch_tensor = tf.train.shuffle_batch(
#         [line_tensor],
#         batch_size=12,
#         capacity=30,
#         min_after_dequeue=12
#     )
#     sess = tf.Session()
#     coord = tf.train.Coordinator()
#     threads = tf.train.start_queue_runners(sess=sess, coord=coord)
#     lines = sess.run(train_data_batch_tensor)
#     for i, line in enumerate(lines):
#         print(line)
#     try:
#         for i in range(1, 2):
#             label, indices, sparse_indices, weight_list, read_count = read_batch(sess, train_data_line, 10)
#             print (label)
#     except tf.errors.OutOfRangeError:
#         print 'Done training -- epoch limit reached'
#     finally:
#         coord.request_stop()
#     coord.join(threads)
#     sess.close()
#
#
# if __name__ == '__main__':
#     main()
