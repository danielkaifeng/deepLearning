import numpy as np
import tensorflow as tf
#from hyper_parameters import *


hyper_para = {
			"weight_decay2":0.0002,
			"weight_decay":0.001
			}


def activation_summary(x):
	#return: Add histogram summary and scalar summary of the sparsity of the tensor
	tensor_name = x.op.name
	tf.summary.histogram(tensor_name + '/activations', x)
	tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


def create_variables(name, shape, initializer=tf.contrib.layers.xavier_initializer(), is_fc_layer=False):
	## TODO: to allow different weight decay to fully connected layer and conv layer

	if is_fc_layer is True:
		regularizer = tf.contrib.layers.l2_regularizer(scale=hyper_para["weight_decay"])
	else:
		regularizer = tf.contrib.layers.l2_regularizer(scale=hyper_para["weight_decay"])

	new_variables = tf.get_variable(name, shape=shape, initializer=initializer,
									regularizer=regularizer)
	return new_variables


def output_layer(input_layer, num_labels):
	'''
	:param input_layer: 2D tensor
	:param num_labels: int. How many output labels in total? (10 for cifar10 and 100 for cifar100)
	:return: output layer Y = WX + B
	'''
	input_dim = input_layer.get_shape().as_list()[-1]
	fc_w = create_variables(name='fc_weights', shape=[input_dim, num_labels], is_fc_layer=True,
							initializer=tf.uniform_unit_scaling_initializer(factor=1.0))
	fc_b = create_variables(name='fc_bias', shape=[num_labels], initializer=tf.zeros_initializer())

	fc_h = tf.matmul(input_layer, fc_w) + fc_b
	return fc_h


def conv_bn_relu_layer(input_layer, filter_shape, stride):
	'''
	A helper function to conv, batch normalize and relu the input tensor sequentially
	:param input_layer: 4D tensor
	:param filter_shape: list. [filter_height, filter_width, filter_depth, filter_number]
	:param stride: stride size for conv
	:return: 4D tensor. Y = Relu(batch_normalize(conv(X)))
	'''

	out_channel = filter_shape[-1]
	filter = create_variables(name='conv', shape=filter_shape)

	conv_layer = tf.nn.conv1d(input_layer, filter, stride=stride, padding='SAME')
	bn_layer = batch_normalization_layer(conv_layer, out_channel)

	output = tf.nn.relu(bn_layer)
	return output


def residual_block(input_layer, output_channel,is_train):
	input_channel = input_layer.get_shape().as_list()[-1]

	if input_channel * 2 == output_channel:
		increase_dim = True
		stride = 2
	elif input_channel == output_channel:
		increase_dim = False
		stride = 1
	else:
		raise ValueError('Output and input channel does not match in residual blocks!!!')


#	bn_layer = tf.layers.batch_normalization(input_layer,axis=-1,momentum=0.99,epsilon=0.001,training=is_train)
	if tf.equal(is_train,tf.constant(True)) is not None:
		TF = True
	else:
		TF = False
	bn_layer = tf.contrib.layers.batch_norm(input_layer,is_training=TF)
	with tf.variable_scope('conv1_in_block'):
		conv1 = tf.layers.conv1d(bn_layer,filters=output_channel,kernel_size=5,padding="same",activation=tf.nn.tanh,strides=stride)

	with tf.variable_scope('conv2_in_block'):
		conv2 = tf.layers.conv1d(conv1,filters=output_channel,kernel_size=5,padding="same",activation=tf.nn.tanh,strides=1)

	if increase_dim is True:
		pooled_input = tf.layers.average_pooling1d(input_layer, pool_size=2,strides=2, padding='VALID')
		padded_input = tf.pad(pooled_input, [[0, 0], [0, 0], [input_channel // 2, input_channel // 2]])
	else:
		padded_input = input_layer

	output = conv2 + padded_input
	return output

def cnn_ectraction(x, drop,n_input,is_train,n_classes):
	input_layer = tf.reshape(x, [-1, n_input, 1])
	regularizer = tf.contrib.layers.l2_regularizer(scale=0.1)

	conv1_1 = tf.layers.conv1d(inputs=input_layer,filters=80,kernel_size=11,
			kernel_regularizer = regularizer,
			bias_regularizer = regularizer,
			padding="same",	activation=tf.nn.tanh,name='conv1')
	conv1_2 = tf.layers.conv1d(inputs=conv1_1,filters=80,kernel_size=11,
			kernel_regularizer = regularizer,
			bias_regularizer = regularizer,
			padding="same",	activation=tf.nn.tanh)
	pool1 = tf.layers.max_pooling1d(inputs=conv1_2, pool_size=4, strides=4)

	conv2_1 = tf.layers.conv1d(inputs=pool1,filters=30,kernel_size=13,padding="same",activation=tf.nn.tanh,name='conv2')
	conv2_2 = tf.layers.conv1d(inputs=conv2_1,filters=30,kernel_size=13,padding="same",activation=tf.nn.tanh)
	pool2_1 = tf.layers.max_pooling1d(inputs=conv2_2, pool_size=4, strides=4)

	num_channel = pool2_1.get_shape().as_list()[-1]

	layers = [pool2_1]
	for ri in range(4):
		with tf.variable_scope('conv_res_1%d' % ri, reuse=False):
			res_conv = residual_block(layers[-1], num_channel,is_train)
			layers.append(res_conv)

	for ri in range(4):
		with tf.variable_scope('conv_res_2%d' % ri, reuse=False):
			res_conv = residual_block(layers[-1], num_channel*2,is_train)
			layers.append(res_conv)

	for ri in range(4):
		with tf.variable_scope('conv_res_3%d' % ri, reuse=False):
			res_conv = residual_block(layers[-1], num_channel*4,is_train)
			layers.append(res_conv)
	res_layer = layers[-1]
#	activation_summary(pool222)
#	assert conv3.get_shape().as_list()[1:] == [8, 8, 64]
	
	dropout2 = tf.layers.dropout(inputs=res_layer, rate=drop)

	conv3_1 = tf.layers.conv1d(inputs=dropout2,filters=20,kernel_size=13,padding="same",activation=tf.nn.tanh,name='conv3_1')
	conv3_2 = tf.layers.conv1d(inputs=conv3_1,filters=20,kernel_size=13,padding="same",activation=tf.nn.tanh,name='conv3_2')
	pool3 = tf.layers.max_pooling1d(inputs=conv3_2, pool_size=4, strides=4)

	dropout3 = tf.layers.dropout(inputs=pool3, rate=drop)
	
	conv4_1 = tf.layers.conv1d(inputs=dropout3,filters=20,kernel_size=11,padding="same",activation=tf.nn.tanh)
	conv4_2 = tf.layers.conv1d(inputs=conv4_1,filters=20,kernel_size=11,padding="same",activation=tf.nn.tanh)
	pool4 = tf.layers.max_pooling1d(inputs=conv4_2, pool_size=4, strides=4)
	
	dropout4 = tf.layers.dropout(inputs=pool4, rate=drop)

	pool_flat = tf.reshape(dropout3, [-1, int((n_input/256) * 20) ])

	dense1 = tf.layers.dense(inputs=pool_flat, units=50, activation=tf.nn.tanh)
	dropout5 = tf.layers.dropout(inputs=dense1, rate=drop)
	
	dense2 = tf.layers.dense(inputs=dropout5, units=n_classes)
	logit = tf.nn.softmax(dense2)

	kernel1 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'conv1/kernel')[0]
	kernel2 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'conv2/kernel')[0]
	kernel3_1 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'conv3_1/kernel')[0]
	kernel3_2 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'conv3_2/kernel')[0]

	return logit,dense1,(kernel1,kernel2,kernel3_1,kernel3_2)




def inference(input_tensor_batch, n, reuse):
	'''
	The main function that defines the ResNet. total layers = 1 + 2n + 2n + 2n +1 = 6n + 2
	:param input_tensor_batch: 4D tensor
	:param n: num_residual_blocks
	:param reuse: To build train graph, reuse=False. To build validation graph and share weights
	with train graph, resue=True
	:return: last layer in the network. Not softmax-ed
	'''

	layers = []
	with tf.variable_scope('conv0', reuse=reuse):
		conv0 = conv_bn_relu_layer(input_tensor_batch, [3, 3, 3, 16], 1)
		activation_summary(conv0)
		layers.append(conv0)

	for i in range(n):
		with tf.variable_scope('conv1_%d' %i, reuse=reuse):
			if i == 0:
				conv1 = residual_block(layers[-1], 16, first_block=True)
			else:
				conv1 = residual_block(layers[-1], 16)
			activation_summary(conv1)
			layers.append(conv1)

	for i in range(n):
		with tf.variable_scope('conv2_%d' %i, reuse=reuse):
			conv2 = residual_block(layers[-1], 32)
			activation_summary(conv2)
			layers.append(conv2)

	for i in range(n):
		with tf.variable_scope('conv3_%d' %i, reuse=reuse):
			conv3 = residual_block(layers[-1], 64)
			layers.append(conv3)
		assert conv3.get_shape().as_list()[1:] == [8, 8, 64]

	with tf.variable_scope('fc', reuse=reuse):
		in_channel = layers[-1].get_shape().as_list()[-1]
		bn_layer = batch_normalization_layer(layers[-1], in_channel)
		relu_layer = tf.nn.relu(bn_layer)
		global_pool = tf.reduce_mean(relu_layer, [1, 2])

		assert global_pool.get_shape().as_list()[-1:] == [64]
		output = output_layer(global_pool, 10)
		layers.append(output)

	return layers[-1]


def test_graph(train_dir='logs'):
	'''
	Run this function to look at the graph structure on tensorboard. A fast way!
	:param train_dir:
	'''
	input_tensor = tf.constant(np.ones([128, 32, 3]), dtype=tf.float32)
	result = inference(input_tensor, 2, reuse=False)
	init = tf.initialize_all_variables()
	sess = tf.Session()
	sess.run(init)
	summary_writer = tf.train.SummaryWriter(train_dir, sess.graph)
