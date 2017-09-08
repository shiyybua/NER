# -*- coding: utf-8 -*
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from tensorflow.contrib.rnn import DropoutWrapper
from utils import *


BATCH_SIZE = 128
unit_num = embeddings_size         # 默认词向量的大小等于RNN(每个time step) 和 CNN(列) 中神经单元的个数, 为了避免混淆model中全部用unit_num表示。
time_step = max_sequence      # 每个句子的最大长度和time_step一样,为了避免混淆model中全部用time_step表示。
DROPOUT_RATE = 0.6
EPOCH = 30000
TAGS_NUM = get_class_size()


class NER_net:
    def __init__(self, scope_name, iterator, embedding, batch_size):
        self.batch_size = batch_size
        self.embedding = embedding
        self.iterator = iterator
        with tf.variable_scope(scope_name) as scope:
            self._build_net()

    def _build_net(self):
        self.global_step = tf.Variable(0, trainable=False)
        source = self.iterator.source
        tgt = self.iterator.target_input
        max_sequence_in_batch = self.iterator.source_sequence_length
        max_sequence_in_batch = tf.reduce_max(max_sequence_in_batch)
        max_sequence_in_batch = tf.to_int32(max_sequence_in_batch)

        # x: [batch_size, time_step, embedding_size], float32
        self.x = tf.nn.embedding_lookup(self.embedding, source)
        # y: [batch_size, time_step]
        self.y = tgt

        cell_forward = tf.contrib.rnn.BasicLSTMCell(unit_num)
        cell_backward = tf.contrib.rnn.BasicLSTMCell(unit_num)
        if DROPOUT_RATE is not None:
            cell_forward = DropoutWrapper(cell_forward, input_keep_prob=1.0, output_keep_prob=DROPOUT_RATE)
            cell_backward = DropoutWrapper(cell_backward, input_keep_prob=1.0, output_keep_prob=DROPOUT_RATE)

        # time_major 可以适应输入维度。
        outputs, bi_state = \
            tf.nn.bidirectional_dynamic_rnn(cell_forward, cell_backward, self.x, dtype=tf.float32)

        forward_out, backward_out = outputs
        outputs = tf.concat([forward_out, backward_out], axis=2)

        # projection:
        W = tf.get_variable("projection_w", [2 * unit_num, TAGS_NUM])
        b = tf.get_variable("projection_b", [TAGS_NUM])
        x_reshape = tf.reshape(outputs, [-1, 2 * unit_num])
        projection = tf.matmul(x_reshape, W) + b

        # -1 to time step
        self.outputs = tf.reshape(projection, [self.batch_size, -1, TAGS_NUM])

        self.seq_length = tf.convert_to_tensor(self.batch_size * [max_sequence_in_batch], dtype=tf.int32)
        self.log_likelihood, self.transition_params = tf.contrib.crf.crf_log_likelihood(
            self.outputs, self.y, self.seq_length)

        # Add a training op to tune the parameters.
        self.loss = tf.reduce_mean(-self.log_likelihood)
        self.train_op = tf.train.AdamOptimizer().minimize(self.loss)


def train(net, iterator, sess):
    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state(model_path)
    if ckpt != None:
        path = ckpt.model_checkpoint_path
        print 'loading pre-trained model from %s.....' % path
        saver.restore(sess, path)

    current_epoch = sess.run(net.global_step)
    while True:
        if current_epoch > EPOCH: break
        try:
            tf_unary_scores, tf_transition_params, _, losses = sess.run(
                [net.outputs, net.transition_params, net.train_op, net.loss])

            if current_epoch % 100 == 0:
                print '*' * 100
                print current_epoch, 'loss', losses
                print '*' * 100

            if current_epoch % (EPOCH / 10) == 0 and current_epoch != 0:
                sess.run(tf.assign(net.global_step, current_epoch))
                saver.save(sess, model_path+'points', global_step=current_epoch)


            current_epoch += 1

        except tf.errors.OutOfRangeError:
            sess.run(iterator.initializer)
        except tf.errors.InvalidArgumentError:
            print current_epoch, ' iterator.next() cannot get enough data to a batch, initialize it.'
            sess.run(iterator.initializer)
    print 'training finished!'


def predict(net, iterator, sess):
    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state(model_path)
    if ckpt != None:
        path = ckpt.model_checkpoint_path
        print 'loading pre-trained model from %s.....' % path
        saver.restore(sess, path)
    else:
        print 'Model not found, please train your model first'
        return
    while True:
        # batch等于1的时候本来就没有padding，如果批量预测的话，记得这里需要做长度的截取。
        try:
            tf_unary_scores, tf_transition_params = sess.run(
                [net.outputs, net.transition_params])
        except tf.errors.OutOfRangeError:
            print 'Prediction finished!'
            break

        # 把batch那个维度去掉
        tf_unary_scores = np.squeeze(tf_unary_scores)

        viterbi_sequence, _ = tf.contrib.crf.viterbi_decode(
            tf_unary_scores, tf_transition_params)

        for id in viterbi_sequence:
            print id,
        print



if __name__ == '__main__':
    action = 'train'
    vocab_size = get_src_vocab_size()
    src_unknown_id = tgt_unknown_id = vocab_size
    src_padding = vocab_size + 1

    src_vocab_table, tgt_vocab_table = create_vocab_tables(src_vocab_file, tgt_vocab_file, src_unknown_id,
                                                           tgt_unknown_id)
    embedding = load_word2vec_embedding(vocab_size)

    if action == 'train':
        iterator = get_iterator(src_vocab_table, tgt_vocab_table, vocab_size, BATCH_SIZE)
    elif action == 'predict':
        BATCH_SIZE = 1
        iterator = get_predict_iterator(src_vocab_table, vocab_size, BATCH_SIZE)

    net = NER_net("ner", iterator, embedding, BATCH_SIZE)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(iterator.initializer)
        tf.tables_initializer().run()

        if action == 'train':
            train(net, iterator, sess)
        elif action == 'predict':
            pass




