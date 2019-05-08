import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import time
import sys

def octconv(input_lf,input_hf,alpha_out,out_channels,scope='octconv'):
	with tf.variable_scope(scope):
		out_channels_lf = int(alpha_out*out_channels)
		out_channels_hf = out_channels-out_channels_lf 

		output_lf_1 = slim.conv2d(input_lf,out_channels_lf,[3,3],padding='SAME',activation_fn=None,scope='conv_l2l')
		output_hf_1 = slim.conv2d(input_hf,out_channels_hf,[3,3],padding='SAME',activation_fn=None,scope='conv_h2h')

		hf_pool = slim.max_pool2d(input_hf,[2,2],stride=2,padding='SAME',scope='pool')
		output_lf_2 = slim.conv2d(hf_pool,out_channels_lf,[3,3],padding='SAME',activation_fn=None,scope='conv_h2l')

		hf_ups = slim.conv2d(input_lf,out_channels_hf,[3,3],padding='SAME',activation_fn=None,scope='conv_l2h')
		output_hf_2 = tf.image.resize_images(hf_ups,(input_hf.shape[1].value,input_hf.shape[2].value),1)

		output_lf = output_lf_1+output_lf_2
		output_hf = output_hf_1+output_hf_2

		return output_lf,output_hf 

def orig_conv(input,out_channels,scope='orig_conv'):
	return slim.conv2d(input,out_channels,[3,3],padding='SAME',activation_fn=None,scope=scope)



if __name__ == '__main__':
	orig_time = []
	oct_time = []
	alpha_list = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
	channels_list = [20,40,100,200]

	if len(sys.argv)>1:
		alpha0  = alpha_list[int(sys.argv[1])]
		channels = channels_list[int(sys.argv[2])]
	else:
		alpha0 = 0.5
		channels = 100

	with tf.Graph().as_default(),tf.Session() as sess:
		channels_lf = int(alpha0*channels)
		channels_hf = channels-channels_lf

		input_lf_plh = tf.placeholder(tf.float32,[None,32,32,channels_lf])
		input_hf_plh = tf.placeholder(tf.float32,[None,64,64,channels_hf])
		input_orig_plh = tf.placeholder(tf.float32,[None,64,64,channels])

		input_lf = np.ones((20,32,32,channels_lf))
		input_hf = np.ones((20,64,64,channels_hf))
		input_orig = np.ones((20,64,64,channels))

		l1,h1 = octconv(input_lf_plh,input_hf_plh,0.5,40)
		orig = orig_conv(input_orig_plh,40)

		null = slim.conv2d(input_orig,30,[3,3],padding='SAME',activation_fn=None,scope='null')

		sess.run(tf.global_variables_initializer())

		sess.run(null)

		feed_dict2 = {input_orig_plh:input_orig}
		time_start = time.time()
		sess.run([orig],feed_dict=feed_dict2)
		time_end = time.time()
		orig_time = time_end-time_start

		feed_dict1 = {input_lf_plh:input_lf,input_hf_plh:input_hf}
		time_start = time.time()
		sess.run([l1,h1],feed_dict=feed_dict1)
		time_end = time.time()
		oct_time = time_end-time_start

	print('alpha:',alpha0,'orig_conv time:',orig_time,'octconv time:',oct_time)