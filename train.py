# -*- coding: utf-8 -*-
"""
@author : Haoran You

"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('../tf-learning/mnist', fake_data=False)

learning_rate = 0.001
epochs = 3
batch_size = 100
display_iter = 100
quantize = True

g = tf.Graph()

with g.as_default():
    # data (images, labels) defination
    x = tf.placeholder(tf.float32, shape=(batch_size, 784), name='input_x')
    y = tf.placeholder(tf.int64, shape=(batch_size,), name='input_y')
    # model defination
    with tf.name_scope('layer_1'):
        weight = tf.Variable(tf.truncated_normal([784, 512], stddev=0.1))
        bias = tf.Variable(tf.constant(0.1, shape=[512]))
        activation = tf.nn.relu
        pred = activation(tf.matmul(x, weight) + bias)
    with tf.name_scope('layer_2'):
        weight = tf.Variable(tf.truncated_normal([512, 10], stddev=0.1))
        bias = tf.Variable(tf.constant(0.1, shape=[10]))
        pred = tf.matmul(pred, weight) + bias
    # calculate loss
    cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels=y, logits=pred)
    tf.summary.scalar('cross_entropy', cross_entropy)
    # calculate accuracy
    correct_pred = tf.equal(tf.argmax(pred, 1), y)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    tf.summary.scalar('accuracy', accuracy)
    # quantize model
    if quantize:
        tf.contrib.quantize.create_training_graph(quant_delay=0)
    # define optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
    # summaries
    merged = tf.summary.merge_all()
    # initialized all variables
    init_op = tf.global_variables_initializer()

with tf.Session(graph=g) as sess:
    sess.run(init_op)
    train_writer = tf.summary.FileWriter("./logs/train", sess.graph)
    test_writer = tf.summary.FileWriter("./logs/test", sess.graph)
    for epoch in range(epochs):
        batch_num = int(mnist.train.num_examples/batch_size)
        for i in range(batch_num):
            x_train, y_train = mnist.train.next_batch(batch_size, fake_data=False)
            x_train = x_train * 2 - 1.0
            summ, _ = sess.run([merged, optimizer], feed_dict={x:x_train, y:y_train})
            train_writer.add_summary(summ, epoch*batch_num+1)
            if (i % display_iter == 0):
                acc = sess.run(accuracy, feed_dict={x:x_train, y:y_train})
                print("Epoch: {0} Iter: {1} acc: {2:.2f}".format(epoch, i, acc))
        # validate on test set
        batch_num = int(mnist.test.num_examples/batch_size)
        ave_acc = 0.
        for j in range(batch_num):
            x_test, y_test = mnist.test.next_batch(batch_size)
            summ, _, acc = sess.run([merged, optimizer, accuracy], feed_dict={x:x_test, y:y_test})
            ave_acc += acc
        ave_acc = ave_acc / batch_num
        print("Testset Accuracy at epoch %s: [%s]" % (epoch, ave_acc))
    train_writer.close()
    test_writer.close()

    tf.io.write_graph(sess.graph_def, "./logs", "quan_train_model.pb", as_text=False)
    saver = tf.train.Saver()
    saver.save(sess, save_path="./logs/model.ckpt")

g_eval = tf.Graph()

with g_eval.as_default():
    # data (images, labels) defination
    x = tf.placeholder(tf.float32, shape=(batch_size, 784), name='input_x')
    y = tf.placeholder(tf.int64, shape=(batch_size,), name='input_y')
    # model defination
    with tf.name_scope('layer_1'):
        weight = tf.Variable(tf.truncated_normal([784, 512], stddev=0.1))
        bias = tf.Variable(tf.constant(0.1, shape=[512]))
        activation = tf.nn.relu
        pred = activation(tf.matmul(x, weight) + bias)
    with tf.name_scope('layer_2'):
        weight = tf.Variable(tf.truncated_normal([512, 10], stddev=0.1))
        bias = tf.Variable(tf.constant(0.1, shape=[10]))
        pred = tf.matmul(pred, weight) + bias
    # calculate loss
    cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels=y, logits=pred)
    tf.summary.scalar('cross_entropy', cross_entropy)
    # calculate accuracy
    correct_pred = tf.equal(tf.argmax(pred, 1), y)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    tf.summary.scalar('accuracy', accuracy)
    # quantize model
    if quantize:
        tf.contrib.quantize.create_eval_graph()
    # summaries
    merged = tf.summary.merge_all()
    # initialized all variables
    init_op = tf.global_variables_initializer()

input_node_name = ['input_x']
output_node_name = ['layer_2/act_quant/FakeQuantWithMinMaxVars']

with tf.Session(graph=g_eval) as sess:
    sess.run(init_op)
    saver = tf.train.Saver()
    saver.restore(sess, "./logs/model.ckpt")
    frozen_graph_def = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def, output_node_name)
    with open("./frozen_graph.pb", "wb") as f:
        f.write(frozen_graph_def.SerializeToString())
        f.close()
    sess.close()
    # tf.io.write_graph(sess.graph_def, "./logs", "eval_model.pb", as_text=False)