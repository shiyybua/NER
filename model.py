# -*- coding: utf-8 -*
import tensorflow as tf
import numpy as np
from tensorflow.contrib.rnn import DropoutWrapper
import utils


BATCH_SIZE = 128
unit_num = utils.embeddings_size         # 默认词向量的大小等于RNN(每个time step) 和 CNN(列) 中神经单元的个数, 为了避免混淆model中全部用unit_num表示。
time_step = utils.max_sequence      # 每个句子的最大长度和time_step一样,为了避免混淆model中全部用time_step表示。
DROPOUT_RATE = 0.5
EPOCH = 60000
TAGS_NUM = utils.get_class_size()


class NER_net:
    def __init__(self, scope_name, iterator, embedding, batch_size):
        self.batch_size = batch_size
        self.embedding = embedding
        self.iterator = iterator
        with tf.variable_scope(scope_name) as scope:
            self._build_net()

    def _build_net(self):
        source = self.iterator.source
        tgt = self.iterator.tgt_input_ids

        # x: [batch_size, time_step, embedding_size], float32
        self.x = tf.nn.embedding_lookup(self.embedding, source)
        # y: [batch_size, time_step]
        self.y = tgt
        # seq_x = tf.reshape(self.x, [-1, time_step * unit_num])
        # seq_x = tf.split(seq_x, time_step, axis=1)

        cell_forward = tf.contrib.rnn.BasicLSTMCell(unit_num)
        cell_backward = tf.contrib.rnn.BasicLSTMCell(unit_num)
        if DROPOUT_RATE is not None:
            cell_forward = DropoutWrapper(cell_forward, input_keep_prob=1.0, output_keep_prob=DROPOUT_RATE)
            cell_backward = DropoutWrapper(cell_backward, input_keep_prob=1.0, output_keep_prob=DROPOUT_RATE)

        # sequence_length=sequence_length, time_major=self.time_major may be needed.
        outputs, output_state_fw, output_state_bw = \
            tf.nn.bidirectional_dynamic_rnn(cell_forward, cell_backward, self.x, dtype=tf.float32)

        rnn_features = tf.transpose(outputs, [1, 0, 2])
        rnn_features = tf.reshape(rnn_features, [-1, 2 * unit_num])

        # CNN
        # You could use more advanced kernel, which is introduced in
        # https://github.com/tensorflow/models/blob/master/tutorials/image/cifar10/cifar10.py
        filter_deep = 2  # 推荐2 或 1   # 值越大，CNN特征占的比例就越多。 如果是2表示CNN的特征和bi-rnn的特征数量一样。
        cnn_W = tf.get_variable("cnn_w", shape=[time_step, 3, 1, filter_deep])
        cnn_b = tf.get_variable("cnn_b", shape=[filter_deep])
        # it is better to make the units number equal to the RNN unit number
        # cnn_input : (batch_size, time_step, unit_num, 1)
        cnn_input = tf.expand_dims(self.x, axis=3)
        # conv_features : (batch_size, time_step, unit_num, 2)
        conv_features = tf.nn.conv2d(cnn_input, cnn_W, strides=[1, 1, 1, 1], padding='SAME') + cnn_b
        if DROPOUT_RATE is not None:
            conv_features = tf.nn.dropout(conv_features, keep_prob=DROPOUT_RATE)
        conv_features = tf.reshape(conv_features, [-1, unit_num * filter_deep])

        all_feature = tf.concat([rnn_features, conv_features], axis=1)

        # projection:
        W = tf.get_variable("projection_w", [(filter_deep + 2) * unit_num, TAGS_NUM])  # 这里的2是指bi-rnn，所以是个常量
        b = tf.get_variable("projection_b", [TAGS_NUM])
        projection = tf.matmul(all_feature, W) + b

        self.outputs = tf.reshape(projection, [-1, time_step, TAGS_NUM])
        # self.outputs = tf.transpose(output, [1,0,2]) #BATCH_SIZE * time_step * TAGS_NUM

        self.log_likelihood, self.transition_params = tf.contrib.crf.crf_log_likelihood(
            self.outputs, self.y, np.array(self.batch_size * [time_step]))

        # Add a training op to tune the parameters.
        self.loss = tf.reduce_mean(-self.log_likelihood)
        self.train_op = tf.train.AdamOptimizer().minimize(self.loss)