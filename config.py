# -*- coding: utf-8 -*-
import tensorflow as tf

tf.app.flags.DEFINE_string("src_file", 'resource/source.txt', "Training data.")
tf.app.flags.DEFINE_string("tgt_file", 'resource/target.txt', "labels.")
# 希望做命名识别的数据
tf.app.flags.DEFINE_string("pred_file", 'resource/predict.txt', "test data.")
tf.app.flags.DEFINE_string("src_vocab_file", 'resource/source_vocab.txt', "source vocabulary.")
tf.app.flags.DEFINE_string("tgt_vocab_file", 'resource/target_vocab.txt', "targets.")
tf.app.flags.DEFINE_string("word_embedding_file", '/Users/mac/Desktop/wiki.zh/wiki.zh.vec', "extra word embeddings.")
tf.app.flags.DEFINE_string("model_path", 'resource/model/', "model save path")
# 这里默认词向量的维度是300, 如果自行训练的词向量维度不是300,则需要该这里的值。
tf.app.flags.DEFINE_integer("embeddings_size", 300, "Size of word embedding.")
tf.app.flags.DEFINE_integer("max_sequence", 100, "max sequence length.")

tf.app.flags.DEFINE_integer("batch_size", 128, "batch size.")
tf.app.flags.DEFINE_integer("epoch", 10000, "epoch.")
tf.app.flags.DEFINE_float("dropout", 0.6, "drop out")

tf.app.flags.DEFINE_string("action", 'predict', "train | predict")
FLAGS = tf.app.flags.FLAGS
