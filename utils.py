# -*- coding: utf-8 -*
import tensorflow as tf
from tensorflow.python.ops import lookup_ops
import numpy as np
import collections
import config
import os

src_file = config.FLAGS.src_file
tgt_file = config.FLAGS.tgt_file
# 只有在预测结果时使用。
pred_file = config.FLAGS.pred_file
src_vocab_file = config.FLAGS.src_vocab_file
tgt_vocab_file = config.FLAGS.tgt_vocab_file
word_embedding_file = config.FLAGS.word_embedding_file
model_path = config.FLAGS.model_path
embeddings_size = config.FLAGS.embeddings_size
max_sequence = config.FLAGS.max_sequence



class BatchedInput(collections.namedtuple("BatchedInput",
                                          ("initializer",
                                           "source",
                                           "target_input",
                                           "source_sequence_length",
                                           "target_sequence_length"))):
  pass


def build_word_index():
    '''
        生成单词列表，并存入文件之中。
    :return:
    '''
    if not os.path.exists(word_embedding_file):
        print 'word embedding file does not exist, please check your file path '
        return

    print 'building word index...'
    if not os.path.exists(src_vocab_file):
        with open(src_vocab_file, 'w') as source:
            f = open(word_embedding_file, 'r')
            for line in f:
                values = line.split()
                word = values[0]  # 取词
                if type(word) is unicode:
                    word = word.encode('utf8')
                source.write(word + '\n')
        f.close()
    else:
        print 'source vocabulary file has already existed, continue to next stage.'

    if not os.path.exists(tgt_vocab_file):
        with open(tgt_file, 'r') as source:
            dict_word = {}
            # with open('source_vocab', 'w') as s_vocab:
            for line in source.readlines():
                line = line.strip()
                if line != '':
                    word_arr = line.split()
                    for w in word_arr:
                        dict_word[w] = dict_word.get(w, 0) + 1

            top_words = sorted(dict_word.items(), key=lambda s: s[1], reverse=True)
            with open(tgt_vocab_file, 'w') as s_vocab:
                for word, frequence in top_words:
                    s_vocab.write(word + '\n')
    else:
        print 'target vocabulary file has already existed, continue to next stage.'

    if not os.path.exists(model_path):
        os.makedirs(model_path)


def get_src_vocab_size():
    '''
    :return: 训练数据中共有多少不重复的词。
    '''
    size = 0
    with open(src_vocab_file, 'r') as vocab_file:
        for content in vocab_file.readlines():
            content = content.strip()
            if content != '':
                size += 1
    return size


def get_class_size():
    '''
        获取命名实体识别类别总数。
    :return:
    '''
    size = 0
    with open(tgt_vocab_file, 'r') as vocab_file:
        for content in vocab_file.readlines():
            content = content.strip()
            if content != '':
                size += 1
    # 最后一个是padding
    return size + 1



def create_vocab_tables(src_vocab_file, tgt_vocab_file, src_unknown_id, tgt_unknown_id, share_vocab=False):
  src_vocab_table = lookup_ops.index_table_from_file(
      src_vocab_file, default_value=src_unknown_id)
  if share_vocab:
    tgt_vocab_table = src_vocab_table
  else:
    tgt_vocab_table = lookup_ops.index_table_from_file(
        tgt_vocab_file, default_value=tgt_unknown_id)
  return src_vocab_table, tgt_vocab_table


def get_iterator(src_vocab_table, tgt_vocab_table, vocab_size, batch_size, buffer_size=None, random_seed=None,
                 num_threads=8, src_max_len=max_sequence, tgt_max_len=max_sequence, num_buckets=5):
    if buffer_size is None:
        # 如果buffer_size比总数据大很多，则会报End of sequence warning。
        # https://github.com/tensorflow/tensorflow/issues/12414
        buffer_size = batch_size * 10

    # src_dataset = tf.contrib.data.TextLineDataset(src_file)
    # tgt_dataset = tf.contrib.data.TextLineDataset(tgt_file)

    src_dataset = tf.data.TextLineDataset(src_file)
    tgt_dataset = tf.data.TextLineDataset(tgt_file)
    src_tgt_dataset = tf.data.Dataset.zip((src_dataset, tgt_dataset))

    src_tgt_dataset = src_tgt_dataset.shuffle(
        buffer_size, random_seed)

    src_tgt_dataset = src_tgt_dataset.map(
        lambda src, tgt: (
            tf.string_split([src]).values, tf.string_split([tgt]).values),
        num_parallel_calls=num_threads)
    src_tgt_dataset.prefetch(buffer_size)

    # src_tgt_dataset = src_tgt_dataset.filter(
    #     lambda src, tgt: tf.logical_and(tf.size(src) > 0, tf.size(tgt) > 0))

    if src_max_len:
        src_tgt_dataset = src_tgt_dataset.map(
            lambda src, tgt: (src[:src_max_len], tgt),
            num_parallel_calls=num_threads)
        src_tgt_dataset.prefetch(buffer_size)
    if tgt_max_len:
        src_tgt_dataset = src_tgt_dataset.map(
            lambda src, tgt: (src, tgt[:tgt_max_len]),
            num_parallel_calls=num_threads)
        src_tgt_dataset.prefetch(buffer_size)

    src_tgt_dataset = src_tgt_dataset.map(
        lambda src, tgt: (tf.cast(src_vocab_table.lookup(src), tf.int32),
                          tf.cast(tgt_vocab_table.lookup(tgt), tf.int32)),
        num_parallel_calls=num_threads)
    src_tgt_dataset.prefetch(buffer_size)

    src_tgt_dataset = src_tgt_dataset.map(
        lambda src, tgt_in: (
            src, tgt_in, tf.size(src), tf.size(tgt_in)),
        num_parallel_calls=num_threads)
    src_tgt_dataset.prefetch(buffer_size)
    def batching_func(x):
        return x.padded_batch(
            batch_size,
            # The first three entries are the source and target line rows;
            # these have unknown-length vectors.  The last two entries are
            # the source and target row sizes; these are scalars.
            padded_shapes=(tf.TensorShape([None]),  # src
                           tf.TensorShape([None]),  # tgt_input
                           tf.TensorShape([]),  # src_len
                           tf.TensorShape([])),  # tgt_len
            # Pad the source and target sequences with eos tokens.
            # (Though notice we don't generally need to do this since
            # later on we will be masking out calculations past the true sequence.
            padding_values=(vocab_size+1,  # src
                            TAG_PADDING_ID,  # tgt_input
                            0,  # src_len -- unused
                            0))

    def key_func(unused_1, unused_2, src_len, tgt_len):
        if src_max_len:
            bucket_width = (src_max_len + num_buckets - 1) // num_buckets
        else:
            bucket_width = 10

        bucket_id = tf.maximum(src_len // bucket_width, tgt_len // bucket_width)
        return tf.to_int64(tf.minimum(num_buckets, bucket_id))

    def reduce_func(unused_key, windowed_data):
        return batching_func(windowed_data)

    batched_dataset = src_tgt_dataset.apply(tf.contrib.data.group_by_window(
        key_func=key_func, reduce_func=reduce_func, window_size=batch_size
    ))
    batched_iter = batched_dataset.make_initializable_iterator()
    (src_ids, tgt_input_ids, src_seq_len, tgt_seq_len) = (
        batched_iter.get_next())

    return BatchedInput(
        initializer=batched_iter.initializer,
        source=src_ids,
        target_input=tgt_input_ids,
        source_sequence_length=src_seq_len,
        target_sequence_length=tgt_seq_len)


def get_predict_iterator(src_vocab_table, vocab_size, batch_size, max_len=max_sequence):
    pred_dataset = tf.contrib.data.TextLineDataset(pred_file)
    pred_dataset = pred_dataset.map(
        lambda src: tf.string_split([src]).values)
    if max_len:
        pred_dataset = pred_dataset.map(lambda src: src[:max_sequence])

    pred_dataset = pred_dataset.map(
        lambda src: tf.cast(src_vocab_table.lookup(src), tf.int32))

    pred_dataset = pred_dataset.map(lambda src: (src, tf.size(src)))

    def batching_func(x):
        return x.padded_batch(
            batch_size,
            padded_shapes=(tf.TensorShape([None]),  # src
                           tf.TensorShape([])),  # src_len
            padding_values=(vocab_size+1,  # src
                            0))  # src_len -- unused

    batched_dataset = batching_func(pred_dataset)
    batched_iter = batched_dataset.make_initializable_iterator()
    (src_ids, src_seq_len) = batched_iter.get_next()

    # 这里target_input在预测的时候不需要，但是不能返回None否则报错。这里则用个placeholder代替，但是仍然不会用到。
    WAHTEVER = 10
    fake_tag = tf.placeholder(tf.int32, [None, WAHTEVER])
    return BatchedInput(
        initializer=batched_iter.initializer,
        source=src_ids,
        target_input=fake_tag,
        source_sequence_length=src_seq_len,
        target_sequence_length=src_seq_len)


def load_word2vec_embedding(vocab_size):
    '''
        加载外接的词向量。
        :return:
    '''
    print 'loading word embedding, it will take few minutes...'
    embeddings = np.random.uniform(-1,1,(vocab_size + 2, embeddings_size))
    # 保证每次随机出来的数一样。
    rng = np.random.RandomState(23455)
    unknown = np.asarray(rng.normal(size=(embeddings_size)))
    padding = np.asarray(rng.normal(size=(embeddings_size)))
    f = open(word_embedding_file)
    for index, line in enumerate(f):
        values = line.split()
        try:
            coefs = np.asarray(values[1:], dtype='float32')  # 取向量
        except ValueError:
            # 如果真的这个词出现在了训练数据里，这么做就会有潜在的bug。那coefs的值就是上一轮的值。
            print values[0], values[1:]

        embeddings[index] = coefs   # 将词和对应的向量存到字典里
    f.close()
    # 顺序不能错，这个和unkown_id和padding id需要一一对应。
    embeddings[-2] = unknown
    embeddings[-1] = padding

    return tf.get_variable("embeddings", dtype=tf.float32,
                           shape=[vocab_size + 2, embeddings_size],
                           initializer=tf.constant_initializer(embeddings), trainable=False)


def tag_to_id_table():
    return lookup_ops.index_to_string_table_from_file(
        tgt_vocab_file, default_value='<tag-unknown>')


def file_content_iterator(file_name):
    with open(file_name, 'r') as f:
        for line in f.readlines():
            yield line.strip()


def write_result_to_file(iterator, tags):
    raw_content = next(iterator)
    words = raw_content.split()
    assert len(words) == len(tags)
    for w,t in zip(words, tags):
        print w, '(' + t + ')',
    print
    print '*' * 100


build_word_index()
TAG_PADDING_ID = get_class_size() - 1




'''
    以下是做测试用的，不用管。
'''
if __name__ == '__main__':
    #################### Just for testing #########################
    vocab_size = get_src_vocab_size()
    src_unknown_id = tgt_unknown_id = vocab_size
    src_padding = vocab_size + 1

    src_vocab_table, tgt_vocab_table = create_vocab_tables(src_vocab_file, tgt_vocab_file, src_unknown_id, tgt_unknown_id)
    # iterator = get_iterator(src_vocab_table, tgt_vocab_table, vocab_size, 100, random_seed=None)
    reverse_tgt_vocab_table = lookup_ops.index_to_string_table_from_file(
        src_vocab_file, default_value='<tag-unknown>')

    iterator = get_predict_iterator(src_vocab_table, vocab_size, 1)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(iterator.initializer)
        tf.tables_initializer().run()

        # 根据ID查字。
        word = reverse_tgt_vocab_table.lookup(tf.constant(12001, dtype=tf.int64))
        print sess.run(word)
        for i in range(10):
            try:
                # source, target = sess.run([iterator.source, iterator.target_input])
                source = sess.run(iterator.source)
                print source.shape, source[0][:5]
                # print i, source.shape, target.shape
            except tf.errors.OutOfRangeError:
                sess.run(iterator.initializer)
                # source, target = sess.run([iterator.source, iterator.target_input])
                source = sess.run(iterator.source)
                print 'new:', source.shape, source[0][:5]


