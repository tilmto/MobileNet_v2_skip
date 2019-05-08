import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

class MobileNet_v2_skip:
	def __init__(self,images_train,labels_train,images_val,labels_val,cr=0.7,early_exit=False,conf_limit=0.6,is_training=True,img_size=256,class_num=1000,scope='MobileNet_v2'):
		self.images_train = images_train
		self.labels_train = labels_train
		self.images_val = images_val
		self.labels_val = labels_val

		self.cr = cr
		self.early_exit = early_exit
		self.is_training = is_training
		self.img_size = img_size
		self.class_num = class_num

		self.images = tf.cond(tf.cast(is_training,tf.bool),lambda:self.images_train,lambda:self.images_val)
		self.labels = tf.cond(tf.cast(is_training,tf.bool),lambda:self.labels_train,lambda:self.labels_val)

		self.lstm_cell=tf.nn.rnn_cell.LSTMCell(num_units=10,num_proj=1,name='lstm_cell')

		self.layer_masks = []
		self.channel_masks = []
		self.if_skip_channel = False

		self.build_model(scope,is_training)

		self.loss_cls = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.predict,labels=self.labels))

		flops = {'conv1':154140672,'16':128974848,'24_1':76283904,'24':67239936,'32_1':40402944,'32':28704768,
				 '64_1':19759104,'64':26935296,'96_1':33226752,'96':59277312,'160_1':40771584,'160':40427520,
				 '320':60088320,'conv2':52428800,'conv3':256000}

		self.flops_fix = flops['conv1']+flops['16']+flops['24_1']+flops['32_1']+flops['64_1']+flops['96_1']+flops['160_1']+flops['320']+flops['conv2']+flops['conv3']
		self.flops_dyn = flops['24']+flops['32']*2+flops['64']*3+flops['96']*2+flops['160']*2
		self.flops_sum = self.flops_fix+self.flops_dyn

		self.computation_cost = tf.reduce_mean(flops['24']*self.layer_masks[0]+flops['32']*tf.reduce_sum(self.layer_masks[1:3],axis=0)
											+flops['64']*tf.reduce_sum(self.layer_masks[3:6],axis=0)
											+flops['96']*tf.reduce_sum(self.layer_masks[6:8],axis=0)
											+flops['160']*tf.reduce_sum(self.layer_masks[8:],axis=0))+self.flops_fix

		self.compress_ratio = self.computation_cost/self.flops_sum

		alpha = 1e-9
		sign = tf.cond(self.compress_ratio>self.cr,lambda:1.0,lambda:-1.0)
		self.loss = self.loss_cls + alpha*sign*self.computation_cost

		self.accuracy = tf.reduce_mean(tf.to_float(tf.equal(tf.argmax(self.predict,axis=1),tf.cast(self.labels,tf.int64))))

		if self.early_exit:
			self.loss_branch_1 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.pred_branch_1,labels=self.labels))
			self.loss_branch_2 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.pred_branch_2,labels=self.labels))
			self.accuracy_branch_1 = tf.reduce_mean(tf.to_float(tf.equal(tf.argmax(self.pred_branch_1,axis=1),tf.cast(self.labels,tf.int64))))
			self.accuracy_branch_2 = tf.reduce_mean(tf.to_float(tf.equal(tf.argmax(self.pred_branch_2,axis=1),tf.cast(self.labels,tf.int64))))
			self.global_loss = self.loss+self.loss_branch_1+self.loss_branch_2

	def build_model(self,scope,is_training=True):
		x = self.images

		with tf.variable_scope(scope):
			x = slim.conv2d(x,32,[7,7],2,padding='SAME',activation_fn=None,scope='conv1')
			x = tf.nn.relu6(self.batch_norm(x,is_training,scope='bn1'))

			x = self.residual(x,out_channels=16,multi=1,stride=1,is_training=self.is_training,scope='residual_16_1')

			x = self.residual(x,out_channels=24,multi=6,stride=2,is_training=self.is_training,scope='residual_24_1')
			next_x = self.residual(x,out_channels=24,multi=6,stride=1,skip_channel=self.if_skip_channel,is_training=self.is_training,scope='residual_24_2')
			mask = self.skip_layer(x,scope='skip_layer_24_1')
			one_tensor = tf.ones_like(mask)
			x = mask*next_x+(one_tensor-mask)*x
			self.layer_masks.append(tf.reshape(mask,[-1]))

			if self.early_exit:
				self.pred_branch_1 = self.branch_classifier(x,scope='branch_1')
				self.prob_array_1 = tf.nn.softmax(self.pred_branch_1,name='prob_array_1')

			x = self.residual(x,out_channels=32,multi=6,stride=2,is_training=self.is_training,scope='residual_32_1')
			for i in range(2):
				next_x = self.residual(x,out_channels=32,multi=6,stride=1,skip_channel=self.if_skip_channel,is_training=self.is_training,scope='residual_32_'+str(i+2))
				mask = self.skip_layer(x,scope='skip_layer_32_'+str(i+1))
				x = mask*next_x+(one_tensor-mask)*x
				self.layer_masks.append(tf.reshape(mask,[-1]))

			x = self.residual(x,out_channels=64,multi=6,stride=2,is_training=self.is_training,scope='residual_64_1')
			for i in range(3):
				next_x = self.residual(x,out_channels=64,multi=6,stride=1,skip_channel=self.if_skip_channel,is_training=self.is_training,scope='residual_64_'+str(i+2))
				mask = self.skip_layer(x,scope='skip_layer_64_'+str(i+1))
				x = mask*next_x+(one_tensor-mask)*x
				self.layer_masks.append(tf.reshape(mask,[-1]))

			x = self.residual(x,out_channels=96,multi=6,stride=1,is_training=self.is_training,scope='residual_96_1')
			for i in range(2):
				next_x = self.residual(x,out_channels=96,multi=6,stride=1,skip_channel=self.if_skip_channel,is_training=self.is_training,scope='residual_96_'+str(i+2))
				mask = self.skip_layer(x,scope='skip_layer_96_'+str(i+1))
				x = mask*next_x+(one_tensor-mask)*x
				self.layer_masks.append(tf.reshape(mask,[-1]))

			if self.early_exit:
				self.pred_branch_2 = self.branch_classifier(x,scope='branch_2')
				self.prob_array_2 = tf.nn.softmax(self.pred_branch_2,name='prob_array_2')

			x = self.residual(x,out_channels=160,multi=6,stride=2,is_training=self.is_training,scope='residual_160_1')
			for i in range(2):
				next_x = self.residual(x,out_channels=160,multi=6,stride=1,skip_channel=self.if_skip_channel,is_training=self.is_training,scope='residual_160_'+str(i+2))
				mask = self.skip_layer(x,scope='skip_layer_160_'+str(i+1))
				x = mask*next_x+(one_tensor-mask)*x
				self.layer_masks.append(tf.reshape(mask,[-1]))

			x = self.residual(x,out_channels=320,multi=6,stride=1,is_training=self.is_training,scope='residual_320_1')

			x = slim.conv2d(x,1280,[1,1],1,padding='SAME',activation_fn=None,scope='conv2')
			x = tf.nn.relu6(self.batch_norm(x,is_training,scope='bn2'))

			x = slim.avg_pool2d(x,[x.shape[1].value,x.shape[2].value],stride=1,padding='VALID',scope='avg_pool')

			x = slim.conv2d(x,self.class_num,[1,1],1,padding='SAME',activation_fn=None,scope='conv3')

			self.predict = tf.reshape(x,[-1,self.class_num],name='predict')


	def residual(self,x,out_channels,multi=6,stride=1,skip_channel=False,is_training=True,scope='residual'):
		in_channels = x.shape[-1].value

		if stride==1:
			orig_x = x

			with tf.variable_scope(scope):
				x = slim.conv2d(x,in_channels*multi,[1,1],1,padding='SAME',activation_fn=None,scope='rconv1')
				x = tf.nn.relu6(self.batch_norm(x,is_training,scope='rbn1'))

				with tf.variable_scope('depthwise_conv'):
					dw_filter = tf.get_variable('dw_filter',[3,3,in_channels*multi,1],initializer=tf.truncated_normal_initializer(stddev=0.1))
					next_x = tf.nn.depthwise_conv2d(x,dw_filter,strides=[1,1,1,1],padding='SAME')
					next_x = tf.nn.relu6(self.batch_norm(next_x,is_training,scope='rbn2'))

				if skip_channel:
					mask = self.skip_channel(x,out_channels=in_channels*multi,scope='skip_channel')
					one_tensor = tf.ones_like(mask)
					x = mask*next_x+(one_tensor-mask)*x
					self.channel_masks.append(mask)
				else:
					x = next_x

				x = slim.conv2d(x,out_channels,[1,1],1,padding='SAME',activation_fn=None,scope='rconv3')

				if in_channels != out_channels:
					orig_x = slim.conv2d(orig_x,out_channels,[1,1],1,padding='SAME',activation_fn=None,scope='orig_conv')

				x = x+orig_x

		else:
			with tf.variable_scope(scope):
				x = slim.conv2d(x,in_channels*multi,[1,1],1,padding='SAME',activation_fn=None,scope='rconv1')
				x = tf.nn.relu6(self.batch_norm(x,is_training,scope='rbn1'))

				with tf.variable_scope('depthwise_conv'):
					dw_filter = tf.get_variable('dw_filter',[3,3,in_channels*multi,1],initializer=tf.truncated_normal_initializer(stddev=0.1))
					x = tf.nn.depthwise_conv2d(x,dw_filter,strides=[1,stride,stride,1],padding='SAME')
					x = tf.nn.relu6(self.batch_norm(x,is_training,scope='rbn2'))

				x = slim.conv2d(x,out_channels,[1,1],1,padding='SAME',activation_fn=None,scope='rconv3')

		return x


	def skip_layer(self,feature_map,scope='skip_layer'):
		x = feature_map
		with tf.variable_scope(scope):
			x = slim.avg_pool2d(x,[x.shape[1].value, x.shape[2].value],stride=1,padding='VALID',scope='avg_pool')
			x = slim.conv2d(x,32,[1,1],1,padding='SAME',activation_fn=None,scope='conv')
			x = tf.reshape(x,[-1,1,32])

			x,state = tf.nn.dynamic_rnn(cell=self.lstm_cell,inputs=x,dtype=tf.float32,scope='rnn')

			x = tf.reshape(x,[-1,1,1,1])
			zero_tensor = tf.zeros_like(x)
			one_tensor = tf.ones_like(x)

			return tf.nn.sigmoid(x)+tf.stop_gradient(tf.where(x>0,one_tensor,zero_tensor)-tf.nn.sigmoid(x))


	def skip_channel(self,feature_map,out_channels,scope='skip_channel'):
		x = feature_map

		with tf.variable_scope(scope):
			x = slim.conv2d(x,x.shape[3].value,[3,3],2,padding='SAME',activation_fn=None,scope='conv')
			x = slim.avg_pool2d(x,[x.shape[1].value, x.shape[2].value],stride=1,padding='VALID',scope='avg_pool')
			x = slim.fully_connected(x,out_channels,scope='fc',activation_fn=None)

			x = tf.reshape(x,[-1,1,1,out_channels])
			zero_tensor = tf.zeros_like(x)
			one_tensor = tf.ones_like(x)

			return tf.nn.sigmoid(x)+tf.stop_gradient(tf.where(x>0,one_tensor,zero_tensor)-tf.nn.sigmoid(x))


	def branch_classifier(self,feature_map,scope='branch'):
		x = feature_map
		with tf.variable_scope(scope):
			x = slim.max_pool2d(x,[2,2],stride=2,padding='SAME',scope='max_pool1')
			x = slim.conv2d(x,300,[3,3],2,padding='SAME',activation_fn=None,scope='conv1')
			x = slim.avg_pool2d(x,[x.shape[1].value,x.shape[2].value],stride=1,padding='VALID',scope='avg_pool')
			x = slim.conv2d(x,self.class_num,[1,1],1,padding='SAME',activation_fn=None,scope='conv2')

			return tf.reshape(x,[-1,self.class_num],name='pred_'+scope)


	def batch_norm(self,x,is_training=True,scope='bn',moving_decay=0.9,eps=1e-6):
		with tf.variable_scope(scope):
			gamma = tf.get_variable('gamma',x.shape[-1],initializer=tf.constant_initializer(1))
			beta  = tf.get_variable('beta', x.shape[-1],initializer=tf.constant_initializer(0))

			axes = list(range(len(x.shape)-1))
			batch_mean, batch_var = tf.nn.moments(x,axes,name='moments')

			ema = tf.train.ExponentialMovingAverage(moving_decay)

			def mean_var_with_update():
				ema_apply_op = ema.apply([batch_mean,batch_var])
				with tf.control_dependencies([ema_apply_op]):
					return tf.identity(batch_mean), tf.identity(batch_var)

			mean, var = tf.cond(tf.equal(is_training,True),mean_var_with_update,
					lambda:(ema.average(batch_mean),ema.average(batch_var)))

			return tf.nn.batch_normalization(x,mean,var,beta,gamma,eps)


	def get_data(self,images,labels,is_training):
		self.images = images
		self.labels = labels
		self.is_training = is_training


def model_size():
	params = tf.trainable_variables()
	size = 0
	for x in params:
		sz = 1
		for dim in x.get_shape():
			sz *= dim.value
		size += sz
	return size


if __name__ == '__main__':
	images = tf.placeholder(tf.float32,[None,256,256,3],name='images')
	labels = tf.placeholder(tf.int32,[None],name ='labels')

	with tf.Session() as sess:
		mobile = MobileNet_v2_skip(images,labels,images,labels)
		sess.run(tf.global_variables_initializer())
		print(mobile.predict.shape.as_list())
		print('Size:',model_size())

		params = tf.trainable_variables()
		for x in params:
			if x.name.find('lstm_cell')!=-1:
				print(x.name)