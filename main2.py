import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import os
import csv
import copy
import time
import data
from MobileNet_v2_orig import MobileNet_v2

def model_size():
    params = tf.trainable_variables()
    size = 0
    for x in params:
        sz = 1
        for dim in x.get_shape():
            sz *= dim.value
        size += sz
    return size

tf.app.flags.DEFINE_string('model','1','1:MobileNet_v2\n2:MobileNet_v2_skip\n')
tf.app.flags.DEFINE_integer('batch_size',20,'Input the batch size')
tf.app.flags.DEFINE_boolean('is_training',True,'Training mode or not')
tf.app.flags.DEFINE_boolean('load_model',False,'Load model or not')
tf.app.flags.DEFINE_boolean('quantize',False,'Quantize or not')
flags = tf.app.flags.FLAGS

config = tf.ConfigProto(inter_op_parallelism_threads=6,intra_op_parallelism_threads=6)
config.gpu_options.allow_growth = True


if __name__ == '__main__':
    is_training_plh = tf.placeholder(tf.bool)

    with tf.Session(config=config) as sess:
        images_train,labels_train = data.batch_q('train',flags.batch_size)
        images_val,labels_val = data.batch_q('val',flags.batch_size)

        if flags.model=='1':
            model = MobileNet_v2(images_train,labels_train,images_val,labels_val,is_training_plh,class_num=200)

        if flags.quantize:
            if flags.is_training:
                tf.contrib.quantize.create_training_graph(quant_delay=0)
            else:
                tf.contrib.quantize.create_eval_graph()

        optimizer = tf.train.AdamOptimizer(1e-4).minimize(model.loss)

        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(max_to_keep=10)

        if not flags.is_training or flags.load_model:
            saver.restore(sess,'my_model_best')
            print('Load model from my_model_best.')
        else:
            print('Create and initialize new model. Size:',model_size())

        max_epoch = 500
        best_accuracy = 0
        best_ep = 0
        train_data_num = 100000
        val_data_num = 10000

        coord = tf.train.Coordinator() 
        threads = tf.train.start_queue_runners(sess=sess,coord=coord)

        if flags.is_training:
            print('do train!!!!!!')
            for ep_idx in range(1,max_epoch):
                train_loss_sum = 0.0
                batch_num = 0

                feed_dict = {is_training_plh:True}

                while True:
                    _,loss = sess.run([optimizer,model.loss],feed_dict=feed_dict)
                    train_loss_sum += loss
                    batch_num += 1

                    if batch_num*flags.batch_size>=train_data_num:
                        break

                print('train_loss_avg:',train_loss_sum/batch_num)

                if not ep_idx%3:
                    accuracy_sum = 0.0
                    batch_num = 0

                    feed_dict = {is_training_plh:False}

                    while True:
                        accuracy_sum += sess.run(model.accuracy,feed_dict=feed_dict)
                        batch_num += 1

                        if batch_num*flags.batch_size>=val_data_num:
                            break

                    accuracy_avg = accuracy_sum/batch_num

                    if accuracy_avg>best_accuracy:
                        best_ep = ep_idx
                        best_accuracy = accuracy_avg
                        saver.save(sess,'./my_model_best')

                    print('epoch: '+ str(ep_idx)+' test accuracy:',accuracy_avg,' best_epoch:',best_ep,' best accuracy: ',best_accuracy)

        else:
            print('do test!!!!!!')
            accuracy_sum = 0.0
            batch_num = 0

            feed_dict = {is_training_plh:False}

            while True:
                accuracy_sum += sess.run(model.accuracy,feed_dict=feed_dict)
                batch_num += 1

                if batch_num*flags.batch_size>=val_data_num:
                    break

            accuracy_avg = accuracy_sum/batch_num
            print('test accuracy:',accuracy_avg)

        coord.request_stop()
        coord.join(threads)
