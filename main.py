import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import os
import csv
import copy
import time
import data
from MobileNet_v2 import MobileNet_v2
from MobileNet_v2_skip import MobileNet_v2_skip

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
tf.app.flags.DEFINE_float('cr',0.7,'Input the compress ratio')
tf.app.flags.DEFINE_boolean('is_training',True,'Training mode or not')
tf.app.flags.DEFINE_boolean('load_model',False,'Load model or not')
tf.app.flags.DEFINE_boolean('load_backbone',False,'Load backbone network only or not')
tf.app.flags.DEFINE_boolean('quantize',False,'Quantize or not')
tf.app.flags.DEFINE_boolean('early_exit',False,'Early exit or not')
tf.app.flags.DEFINE_float('conf',0.6,'Input the confidence threshold for early exit')
flags = tf.app.flags.FLAGS

config = tf.ConfigProto(inter_op_parallelism_threads=6,intra_op_parallelism_threads=6)
config.gpu_options.allow_growth = True


if __name__ == '__main__':
    is_training_plh = tf.placeholder(tf.bool)

    with tf.Session(config=config) as sess:
        images_train,labels_train = data.batch_q('train',flags.batch_size)
        images_val,labels_val = data.batch_q('val',flags.batch_size)

        if flags.model=='1':
            model = MobileNet_v2(images_train,labels_train,images_val,labels_val,flags.early_exit,flags.conf,is_training_plh,class_num=200)
        elif flags.model=='2':
            model = MobileNet_v2_skip(images_train,labels_train,images_val,labels_val,flags.cr,flags.early_exit,flags.conf,is_training_plh,class_num=200)

        if flags.quantize:
            if flags.is_training:
                tf.contrib.quantize.create_training_graph(quant_delay=0)
            else:
                tf.contrib.quantize.create_eval_graph()

        if flags.early_exit:
            loss_tensor = model.global_loss
        else:
            loss_tensor = model.loss
        optimizer = tf.train.AdamOptimizer(1e-4).minimize(loss_tensor)

        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(max_to_keep=10)

        if not flags.is_training or flags.load_model:
            if flags.load_backbone:     
                variables = slim.get_variables_to_restore()
                variables_to_restore = [v for v in variables if v.name.find('skip')==-1 and v.name.find('branch')==-1] 
                saver2 = tf.train.Saver(variables_to_restore,max_to_keep=10)
                saver2.restore(sess,'../best_weight/my_model_best')
            else:
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
                    _,loss = sess.run([optimizer,loss_tensor],feed_dict=feed_dict)
                    train_loss_sum += loss
                    batch_num += 1

                    if batch_num*flags.batch_size>=train_data_num:
                        break

                print('epoch:',ep_idx,' train_loss_avg:',train_loss_sum/batch_num)

                if not ep_idx%3:
                    images,labels = data.batch_q('val',flags.batch_size)
                    model.get_data(images,labels,is_training=False)

                    feed_dict = {is_training_plh:False}

                    if flags.early_exit:
                        accuracy_sum = 0
                        accuracy1_sum = 0
                        accuracy2_sum = 0
                        accuracy_global_sum = 0
                        compress_ratio_sum = 0

                        batch_num = 0

                        while True:
                            accuracy,accuracy1,accuracy2,accuracy_global,compress_ratio = sess.run([model.accuracy,model.accuracy_branch_1,model.accuracy_branch_2,model.accuracy_global,model.compress_ratio],feed_dict=feed_dict)
                            accuracy_sum += accuracy
                            accuracy1_sum += accuracy1
                            accuracy2_sum += accuracy2
                            accuracy_global_sum += accuracy_global
                            compress_ratio_sum += compress_ratio

                            batch_num += 1

                            if batch_num*flags.batch_size>=val_data_num:
                                break

                        accuracy_avg = accuracy_sum/batch_num
                        accuracy1_avg = accuracy1_sum/batch_num
                        accuracy2_avg = accuracy2_sum/batch_num
                        accuracy_global_avg = accuracy_global_sum/batch_num
                        compress_ratio_avg = compress_ratio_sum/batch_num

                        if accuracy_global_avg>best_accuracy:
                            best_ep = ep_idx
                            best_accuracy = accuracy_global_avg
                            saver.save(sess,'./my_model_best')

                        print('accuracy global:',accuracy_global_avg,' compress ratio',compress_ratio_avg,'\nfinal accuracy:',accuracy_avg,' accuracy of branch1:',accuracy1_avg,' accuracy of branch2:',accuracy2_avg,
                            '\nbest global accuracy:',best_accuracy,' best epoch:',best_ep)

                    else:
                        accuracy_sum = 0
                        compress_ratio_sum = 0
                        batch_num = 0

                        while True:
                            accuracy,compress_ratio = sess.run([model.accuracy,model.compress_ratio],feed_dict=feed_dict)
                            accuracy_sum += accuracy
                            compress_ratio_sum += compress_ratio
                            batch_num += 1

                            if batch_num*flags.batch_size>=val_data_num:
                                break

                        accuracy_avg = accuracy_sum/batch_num
                        compress_ratio_avg = compress_ratio_sum/batch_num

                        if accuracy_avg>best_accuracy and compress_ratio_avg<flags.cr+0.03:
                            best_ep = ep_idx
                            best_accuracy = accuracy_avg
                            saver.save(sess,'./my_model_best')

                        print('test accuracy:',accuracy_avg,' compress ratio:',compress_ratio_avg,' best_epoch:',best_ep,' best accuracy:',best_accuracy)

                    images,labels = data.batch_q('train',flags.batch_size)
                    model.get_data(images,labels,is_training=True)

        else:
            print('do test!!!!!!')

            feed_dict = {is_training_plh:False}

            if flags.early_exit:
                accuracy_sum = 0
                accuracy1_sum = 0
                accuracy2_sum = 0
                accuracy_global_sum = 0
                compress_ratio_sum = 0

                batch_num = 0

                while True:
                    accuracy,accuracy1,accuracy2,accuracy_global,compress_ratio = sess.run([model.accuracy,model.accuracy_branch_1,model.accuracy_branch_2,model.accuracy_global,model.compress_ratio],feed_dict=feed_dict)
                    accuracy_sum += accuracy
                    accuracy1_sum += accuracy1
                    accuracy2_sum += accuracy2
                    accuracy_global_sum += accuracy_global
                    compress_ratio_sum += compress_ratio

                    batch_num += 1

                    if batch_num*flags.batch_size>=val_data_num:
                        break

                accuracy_avg = accuracy_sum/batch_num
                accuracy1_avg = accuracy1_sum/batch_num
                accuracy2_avg = accuracy2_sum/batch_num
                accuracy_global_avg = accuracy_global_sum/batch_num
                compress_ratio_avg = compress_ratio_sum/batch_num

                print('accuracy global:',accuracy_global_avg,' compress ratio',compress_ratio_avg,'\nfinal accuracy:',accuracy_avg,' accuracy of branch1:',accuracy1_avg,' accuracy of branch2:',accuracy2_avg)

            else:
                accuracy_sum = 0
                compress_ratio_sum = 0
                batch_num = 0

                while True:
                    accuracy,compress_ratio = sess.run([model.accuracy,model.compress_ratio],feed_dict=feed_dict)
                    accuracy_sum += accuracy
                    compress_ratio_sum += compress_ratio
                    batch_num += 1

                    if batch_num*flags.batch_size>=val_data_num:
                        break

                accuracy_avg = accuracy_sum/batch_num
                compress_ratio_avg = compress_ratio_sum/batch_num

                print('test accuracy:',accuracy_avg,' compress ratio:',compress_ratio_avg)

        coord.request_stop()
        coord.join(threads)
