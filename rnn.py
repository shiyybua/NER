# -*- coding: utf-8 -*
from tensorflow.contrib.rnn import DropoutWrapper
from utils import *


BATCH_SIZE = 128
unit_num = embeddings_size         # 默认词向量的大小等于RNN(每个time step) 和 CNN(列) 中神经单元的个数, 为了避免混淆model中全部用unit_num表示。
time_step = max_sequence      # 每个句子的最大长度和time_step一样,为了避免混淆model中全部用time_step表示。
DROPOUT_RATE = 0.5
EPOCH = 60000
TAGS_NUM = get_class_size()


class NER_net:
    def __init__(self, scope_name, iterator, embedding, batch_size):
        self.batch_size = batch_size
        self.embedding = embedding
        self.iterator = iterator
        with tf.variable_scope(scope_name) as scope:
            self._build_net()

    def _build_net(self):
        source = self.iterator.source
        tgt = self.iterator.target_input
        max_sequence_in_batch = self.iterator.source_sequence_length
        max_sequence_in_batch = tf.reduce_max(max_sequence_in_batch)
        max_sequence_in_batch = tf.to_int32(max_sequence_in_batch)
        # max_sequence_in_batch = tf.constant(100)

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
        outputs, bi_state = \
            tf.nn.bidirectional_dynamic_rnn(cell_forward, cell_backward, self.x, dtype=tf.float32)

        forward_out, backward_out = outputs
        outputs = tf.concat([forward_out, backward_out], axis=2)
        print outputs

        # projection:
        W = tf.get_variable("projection_w", [2 * unit_num, TAGS_NUM])
        b = tf.get_variable("projection_b", [TAGS_NUM])
        x_reshape = tf.reshape(outputs, [-1, 2 * unit_num])
        projection = tf.matmul(x_reshape, W) + b

        # -1 to time step
        output = tf.reshape(projection, [-1, self.batch_size, TAGS_NUM])
        self.outputs = tf.transpose(output, [1, 0, 2])  # BATCH_SIZE * time_step * TAGS_NUM
        print 'outputs:', self.outputs
        print 'y:', self.y
        print max_sequence_in_batch

        self.seq_length = tf.convert_to_tensor(self.batch_size * [max_sequence_in_batch], dtype=tf.int32)
        self.log_likelihood, self.transition_params = tf.contrib.crf.crf_log_likelihood(
            self.outputs, self.y, self.seq_length)

        # Add a training op to tune the parameters.
        self.loss = tf.reduce_mean(-self.log_likelihood)
        self.train_op = tf.train.AdamOptimizer().minimize(self.loss)


if __name__ == '__main__':
    vocab_size = get_src_vocab_size()
    src_unknown_id = tgt_unknown_id = vocab_size
    src_padding = vocab_size + 1

    src_vocab_table, tgt_vocab_table = create_vocab_tables(src_vocab_file, tgt_vocab_file, src_unknown_id,
                                                           tgt_unknown_id)
    iterator = get_iterator(src_vocab_table, tgt_vocab_table, vocab_size, BATCH_SIZE)
    embedding = load_word2vec_embedding(vocab_size)

    net = NER_net("ner", iterator, embedding, BATCH_SIZE)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(iterator.initializer)
        tf.tables_initializer().run()

        for i in range(10000):
            print '*' * 100
            # tf_unary_scores, tf_transition_params, _, losses = sess.run(
            #     [net.outputs, net.transition_params, net.train_op, net.loss])
            # try:
            seq_length, x, y= sess.run(
                [net.seq_length, net.x, net.y])

            print i
            print 'seq_length:',seq_length
            print 'x:',x.shape
            # print 'outputs:',outputs.shape
            print 'y:', y.shape
            # print i, 'loss', losses
            print '*' * 100
            # except Exception, e:
            #     print 'break'
            #     print str(Exception)
            #     print str(e)
            #     print 'seq_length:',seq_length
            #     print 'x:',x.shape
            #     print 'outputs:',outputs.shape
            #     print 'y:', y.shape
            #     break

