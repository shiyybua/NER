# -*- coding: utf-8 -*-
import tensorflow as tf

tf.app.flags.DEFINE_string("src_file", 'resource/source.txt', "Training data.")
tf.app.flags.DEFINE_string("tgt_file", 'resource/target.txt', "labels.")
tf.app.flags.DEFINE_string("pred_file", 'resource/predict.txt', "test data.")
tf.app.flags.DEFINE_string("src_vocab_file", 'resource/source_vocab.txt', "source vocabulary.")
tf.app.flags.DEFINE_string("tgt_vocab_file", 'resource/target_vocab.txt', "targets.")
tf.app.flags.DEFINE_string("word_embedding_file", 'resource/wiki.zh.vec', "extra word embeddings.")
tf.app.flags.DEFINE_string("model_path", 'resource/model/', "model save path")
tf.app.flags.DEFINE_integer("embeddings_size", 300, "Size of word embedding.")
tf.app.flags.DEFINE_integer("max_sequence", 100, "max sequence length.")

tf.app.flags.DEFINE_integer("batch_size", 128, "batch size.")
tf.app.flags.DEFINE_integer("epoch", 30000, "epoch.")
tf.app.flags.DEFINE_float("dropout", 0.6, "drop out")

tf.app.flags.DEFINE_string("action", 'train', "train | predict")
FLAGS = tf.app.flags.FLAGS
