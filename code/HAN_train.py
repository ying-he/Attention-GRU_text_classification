# -*- coding: utf-8 -*-
# Author: HeYing
# Creation Date: 2019-06-09

import tensorflow as tf
import time
import os
import pickle
import numpy as np
from HAN_model import HAN


# Data loading params
# tf.flags.DEFINE_string("yelp_json_path", 'data/yelp_academic_dataset_review.json', "data directory")
# tf.flags.DEFINE_integer("vocab_size", 46960, "vocabulary size")
tf.flags.DEFINE_integer("num_classes", 2, "number of classes")
tf.flags.DEFINE_integer("embedding_size", 400, "Dimensionality of character embedding (default: 200)")
tf.flags.DEFINE_integer("hidden_size", 50, "Dimensionality of GRU hidden layer (default: 50)")
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 5, "Number of training epochs (default: 50)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 3, "Number of checkpoints to store (default: 5)")
tf.flags.DEFINE_integer("max_sent_in_doc", 20, "Number of checkpoints to store (default: 5)")
tf.flags.DEFINE_integer("max_word_in_sent", 40, "Number of checkpoints to store (default: 5)")
tf.flags.DEFINE_integer("evaluate_every", 1000, "evaluate every this many batches")
tf.flags.DEFINE_float("learning_rate", 0.01, "learning rate")
tf.flags.DEFINE_float("grad_clip", 5, "grad clip to prevent gradient explode")


FLAGS = tf.flags.FLAGS


# 载入数据
def load_variable(filename):
    f = open(filename, "rb")
    r = pickle.load(f)
    f.close()
    return r


print("Start loading data...")
# train_x, train_y = load_variable("/Users/heying/Downloads/w2vData/v_trainset")  # 训练集
train_x, train_y = load_variable("./data/v_trainset")  # 训练集 on server
# test_x, test_y = load_variable("/Users/heying/Downloads/w2vData/v_testset")  # 测试集
test_x, test_y = load_variable("~/heying/text_classification/data/v_testset")  # 测试集 on server
# word_embeddings = load_variable("/Users/heying/Downloads/w2vData/word_embedding")  # 词向量字典
word_embeddings = load_variable("~/heying/text_classification/data/word_embedding")  # 词向量字典 on server
# train_x, train_y, dev_x, dev_y = load_dataset(FLAGS.yelp_json_path, FLAGS.max_sent_in_doc, FLAGS.max_word_in_sent)
print("data load finished.")  # 数据载入完毕

with tf.Session() as sess:
    # han = HAN(vocab_size=FLAGS.vocab_size, num_classes=FLAGS.num_classes,
    han = HAN(num_classes=FLAGS.num_classes,
              embedding_size=FLAGS.embedding_size, hidden_size=FLAGS.hidden_size)

    with tf.name_scope('loss'):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=han.input_y,
                                                                      logits=han.out,
                                                                      name='loss'))
    with tf.name_scope('accuracy'):
        predict = tf.argmax(han.out, axis=1, name='predict')
        label = tf.argmax(han.input_y, axis=1, name='label')
        acc = tf.reduce_mean(tf.cast(tf.equal(predict, label), tf.float32))

    timestamp = str(int(time.time()))
    out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
    print("Writing to {}\n".format(out_dir))

    global_step = tf.Variable(0, trainable=False)
    optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
    # RNN中常用的梯度截断，防止出现梯度过大难以求导的现象
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), FLAGS.grad_clip)
    grads_and_vars = tuple(zip(grads, tvars))
    train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

    # Keep track of gradient values and sparsity (optional)
    grad_summaries = []
    for g, v in grads_and_vars:
        if g is not None:
            grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
            grad_summaries.append(grad_hist_summary)

    grad_summaries_merged = tf.summary.merge(grad_summaries)

    loss_summary = tf.summary.scalar('loss', loss)
    acc_summary = tf.summary.scalar('accuracy', acc)

    train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
    train_summary_dir = os.path.join(out_dir, "summaries", "train")
    train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

    dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
    dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
    dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

    checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
    checkpoint_prefix = os.path.join(checkpoint_dir, "model")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

    sess.run(tf.global_variables_initializer())

    def train_step(x_batch, y_batch):
        feed_dict = {
            han.input_x: x_batch,
            han.input_y: y_batch,
            han.max_sentence_num: 20,
            han.max_sentence_length: 40,
            han.batch_size: 64
        }
        _, step, summaries, cost, accuracy = sess.run([train_op, global_step, train_summary_op, loss, acc], feed_dict)

        time_str = str(int(time.time()))
        print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, cost, accuracy))
        train_summary_writer.add_summary(summaries, step)
        return step

    def dev_step(x_batch, y_batch, writer=None):
        feed_dict = {
            han.input_x: x_batch,
            han.input_y: y_batch,
            han.max_sentence_num: 20,
            han.max_sentence_length: 40,
            han.batch_size: 64
        }
        step, summaries, cost, accuracy = sess.run([global_step, dev_summary_op, loss, acc], feed_dict)
        # time_str = str(int(time.time()))
        # print("++++++++++++++dev++++++++++{}: step {}, loss {:g}, acc {:g}".format(time_str, step, cost, accuracy))
        if writer:
            writer.add_summary(summaries, step)
        return accuracy

    for epoch in range(FLAGS.num_epochs):
        print('current epoch %s' % (epoch + 1))
        for i in range(0, 360000, FLAGS.batch_size):
            x_text = train_x[i:i + FLAGS.batch_size]
            y = train_y[i:i + FLAGS.batch_size]
            x = np.zeros((FLAGS.batch_size, FLAGS.max_sent_in_doc, FLAGS.max_word_in_sent, FLAGS.embedding_size))
            for m in range(len(x_text)):
                for n in range(len(x_text[m])):
                    for p in range(len(x_text[m][n])):
                        x[m, n, p] = list(word_embeddings[x_text[m][n][p]])
            step = train_step(x, y)
            if step % FLAGS.evaluate_every == 0:
                accuracy = []
                # dev_step(test_x, test_y, dev_summary_writer)
                for i_test in range(0, 40000, FLAGS.batch_size):
                    xtest_text = test_x[i_test:i_test+FLAGS.batch_size]
                    ytest = test_y[i_test:i_test+FLAGS.batch_size]
                    xtest = np.zeros((FLAGS.batch_size, FLAGS.max_sent_in_doc, FLAGS.max_word_in_sent, FLAGS.embedding_size))
                    for m_test in range(len(xtest_text)):
                        for n_test in range(len(xtest_text[m_test])):
                            for p_test in range(len(xtest_text[m_test][n_test])):
                                xtest[m_test,n_test,p_test] = list(word_embeddings[xtest_text[m_test][n_test][p_test]])
                    batch_accuracy = dev_step(xtest, ytest, dev_summary_writer)
                    accuracy.append(batch_accuracy)
                avg_accuracy = np.mean(accuracy)
                print("++++++++++++++++dev++++++++++++ Average accuracy on step %d is %0.4f." % (step, avg_accuracy))
        # 每个epoch将模型保存下来
        saver.save(sess, checkpoint_prefix, global_step=global_step)


