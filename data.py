 # -*- coding: utf-8 -*-
"""
@author : Haoran You

"""
import glob
import re
import random
import numpy as np
import tensorflow as tf

data_path = '../data/tiny-imagenet-200/'

def build_label_dicts():
    """Build look-up dictionaries for class label and class description

    class labels are 0 to 199 in the same order as tiny-imagenet-200/wnids.txt
    class descriptions are from tiny-imagenet-200/words.txt

    Returns:
    tuple of dicts
        label_dict:
            key: synset (e.g. n01944390)
            value: class integer {0 ... 199}
        class_dict:
            key: class integer {0 ... 199}
            value: text description from words.txt
    """
    label_dict, class_description = {}, {}
    wnids = data_path + 'wnids.txt'
    with open(wnids, 'r') as f:
        for i, line in enumerate(f.readlines()):
            synset = line[:-1] # remove \n
            label_dict[synset] = i
    words = data_path + 'words.txt'
    with open(words, 'r') as f:
        for i, line in enumerate(f.readlines()):
            synset, desc = line.split('\t')
            desc = desc[:-1] # remove \n
            if synset in label_dict:
                class_description[label_dict[synset]] = desc
    return label_dict, class_description

def load_filenames_labels(mode):
    """get filenames and labels

    Args:
        mode: 'train' or 'val'

    Returns:
        list of tuples (jpeg filename with path label)
    """
    label_dict, class_description = build_label_dicts()
    filenames_labels = []
    if mode == 'train':
        names = data_path + 'train/*/images/*.JPEG'
        filenames = glob.glob(names)
        for filename in filenames:
            match = re.search(r'n\d+', filename)
            label = str(label_dict[match.group()])
            filenames_labels.append((filename, label))
    elif mode == 'val':
        val_names = data_path + 'val/val_annotations.txt'
        with open(val_names, 'r') as f:
            for line in f.readlines():
                split_line = line.split('\t')
                filename = data_path + 'val/images/' + split_line[0]
                label = str(label_dict[split_line[1]])
                filenames_labels.append((filename, label))
    return filenames_labels

def read_image(filename_q, mode):
    """Load next jpeg file from filename / label queue
    randomly applies distortions if mode == 'train'.
    standardizes all images.

    Args:
        filename_q: Queue with 2 columns: filename and label
        mode: 'train' or 'val'

    Returns:
        [img, label]:
            img: tf.uint8 tensor [height, width, channels]
            label: tf.uint8 target class label {0 ... 199}
    """
    item = filename_q.dequeue()
    filename = item[0]
    label = item[1]
    file = tf.read_file(filename)
    img = tf.image.decode_jpeg(file, channels=3)
    img = tf.cast(img,tf.float32)
    # image distortion: left/right, random hue, random color saturation
    if mode == 'train':
        #img = tf.random_crop(img, np.array([56, 56, 3]))
        img = tf.image.resize_images(img,(256,256),np.random.randint(4))
        img = (img-127)/128
        img = tf.image.random_flip_left_right(img)
        # img = tf.image.random_hue(img, 0.05)
        #img = tf.image.random_saturation(img, 0.5, 2.0)
    else:
        img = tf.image.resize_images(img,(256,256),np.random.randint(4))
        img = (img-127)/128
        #img = tf.image.crop_to_bounding_box(img, 4, 4, 56, 56)

    label = tf.string_to_number(label, tf.int32)
    #label = tf.cast(label, tf.uint8)

    return [img, label]

def batch_q(mode,batch_size,num_epochs=None):
    """return batch of images using filename queue

    Args:
        mode: train or val
        config: training configuration object

    Returns:
        imgs: tf.uint8 tensor [batch_size, height, width, channels]
        labels: tf.uint8 tensor [batch_size,]
    """
    filenames_labels = load_filenames_labels(mode)
    random.shuffle(filenames_labels)
    filename_q = tf.train.input_producer(filenames_labels,
                                         num_epochs=num_epochs,
                                         shuffle=True)
    # 2 read image threads to keep batch_join queue full
    return tf.train.batch_join([read_image(filename_q, mode) for i in range(2)],
                               batch_size, shapes=[(256,256,3),()],
                               capacity=1024)

if __name__ == '__main__':
    image,label = batch_q('val',20)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        coord = tf.train.Coordinator() 
        threads = tf.train.start_queue_runners(sess=sess,coord=coord)

        img,lab = sess.run([image,label])
        print(img.shape,lab.shape)

        coord.request_stop()
        coord.join(threads)