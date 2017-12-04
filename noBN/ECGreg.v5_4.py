#coding=utf-8
from __future__ import absolute_import
from __future__ import division

import tensorflow as tf
import numpy as np
import pandas as pd
import random
from sys import argv
#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
#sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

from tensorflow.python.framework import graph_util


AF_TRAINING = argv[1]
AF_TEST = argv[2]

n_input = 5120
learning_rate = 0.0002
learning_rate2 = 0.0001
x = tf.placeholder(tf.float32, [None, n_input], name="inputx")
y = tf.placeholder(tf.int32, [None], name="inputy")
keep_prob = tf.placeholder(tf.float32, name="keep_prob")
train_flag = tf.placeholder(tf.bool, name="train_flag")

def read_data(trX,trY,batch_size):
	rg_x = range(trX.shape[0])
	random.shuffle(rg_x)
	x_collection = [np.array(trX[rg_x[x:x+batch_size]]) for x in range(0,len(rg_x),batch_size)]
	y_collection = [np.array(trY[rg_x[x:x+batch_size]]) for x in range(0,len(rg_x),batch_size)]
	
	return x_collection,y_collection


def cnn_ectraction2(x, drop,n_input,is_train):
	input_layer = tf.reshape(x, [-1, n_input, 1])
	regularizer = tf.contrib.layers.l2_regularizer(scale=0.5)

	conv1_1 = tf.layers.conv1d(inputs=input_layer,filters=80,kernel_size=11,
				kernel_regularizer = regularizer,bias_regularizer = regularizer,
				padding="same",	activation=tf.nn.tanh,name='conv1')
	conv1_2 = tf.layers.conv1d(inputs=conv1_1,filters=80,kernel_size=11,
				kernel_regularizer = regularizer,bias_regularizer = regularizer,
				padding="same",	activation=tf.nn.tanh)
	pool1 = tf.layers.max_pooling1d(inputs=conv1_2, pool_size=4, strides=4)

#	bn1 = tf.cond(is_train,
#					lambda: tf.layers.batch_normalization(pool1, training=True,trainable=False),
#					lambda: tf.layers.batch_normalization(pool1, training=False,trainable=False))
	x_mean, x_var = tf.nn.moments(pool1, 0)
	bn1 = tf.nn.batch_normalization(x=pool1, mean=x_mean, variance=x_var, offset=None, scale=None, variance_epsilon=1e-3)

	conv2_1 = tf.layers.conv1d(inputs=bn1,filters=80,kernel_size=15,padding="same",activation=tf.nn.tanh,name='conv2')
	conv2_2 = tf.layers.conv1d(inputs=conv2_1,filters=80,kernel_size=15,padding="same",activation=tf.nn.tanh)
	pool2 = tf.layers.max_pooling1d(inputs=conv2_2, pool_size=2, strides=2)
	
	conv3_1 = tf.layers.conv1d(inputs=pool2,filters=50,kernel_size=11,padding="same",activation=tf.nn.tanh,name='conv3_1')
	conv3_2 = tf.layers.conv1d(inputs=conv3_1,filters=50,kernel_size=11,padding="same",activation=tf.nn.tanh,name='conv3_2')
	pool3 = tf.layers.max_pooling1d(inputs=conv3_2, pool_size=2, strides=2)

	dropout3 = tf.layers.dropout(inputs=pool3, rate=drop)
	
	conv4_1 = tf.layers.conv1d(inputs=dropout3,filters=50,kernel_size=11,padding="same",activation=tf.nn.tanh)
	conv4_2 = tf.layers.conv1d(inputs=conv4_1,filters=50,kernel_size=11,padding="same",activation=tf.nn.tanh)
	pool4 = tf.layers.max_pooling1d(inputs=conv4_2, pool_size=4, strides=4)
	
	dropout4 = tf.layers.dropout(inputs=pool4, rate=drop)

	conv5_1 = tf.layers.conv1d(inputs=dropout4,filters=30,kernel_size=11,padding="same",activation=tf.nn.tanh)
	conv5_2 = tf.layers.conv1d(inputs=conv5_1,filters=30,kernel_size=11,padding="same",activation=tf.nn.tanh)
	pool5 = tf.layers.max_pooling1d(inputs=conv5_2, pool_size=4, strides=4)
	
	pool_flat = tf.reshape(pool5, [-1, int((n_input/256) * 30) ])



	dense1 = tf.layers.dense(inputs=pool_flat, units=50, activation=tf.nn.tanh)

	dropout5 = tf.layers.dropout(inputs=dense1, rate=drop)
	
	dense2 = tf.layers.dense(inputs=dropout5, units=n_classes)
	logit = tf.nn.softmax(dense2)

	kernel1 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'conv1/kernel')[0]
	kernel2 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'conv2/kernel')[0]
	kernel3_1 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'conv2/kernel')[0]
	kernel3_2 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'conv2/kernel')[0]

	return logit,dense1,(kernel1,kernel2,kernel3_1,kernel3_2)

def batch_norm(x, phase_train, scope='bn'):
	"""
	Batch normalization on convolutional maps.
	Args:
		x:		   Tensor, 4D BHWD input maps
		n_out:	   integer, depth of input maps
		phase_train: boolean tf.Varialbe, true indicates training phase
		scope:	   string, variable scope
	Return:
		normed:	  batch-normalized maps
	"""

	n_out = x.get_shape().as_list()[-1]
	with tf.variable_scope(scope):
		beta = tf.Variable(tf.constant(0.0, shape=[n_out]),
									 name='beta', trainable=True)
		gamma = tf.Variable(tf.constant(1.0, shape=[n_out]),
									  name='gamma', trainable=True)
		batch_mean, batch_var = tf.nn.moments(x, [0,1], name='moments')
		ema = tf.train.ExponentialMovingAverage(decay=0.5)

		def mean_var_with_update():
			ema_apply_op = ema.apply([batch_mean, batch_var])
			with tf.control_dependencies([ema_apply_op]):
				return tf.identity(batch_mean), tf.identity(batch_var)

		mean, var = tf.cond(phase_train, mean_var_with_update,
							lambda: (ema.average(batch_mean), ema.average(batch_var)))
		normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
	return normed

def cnn_ectraction(x, drop,n_input,is_train):
	epsilon = 0.001
	input_layer = tf.reshape(x, [-1, n_input, 1])
	regularizer = tf.contrib.layers.l2_regularizer(scale=0.5)

	conv1_1 = tf.layers.conv1d(inputs=input_layer,filters=64,kernel_size=11,
			kernel_regularizer = regularizer,
			bias_regularizer = regularizer,
			padding="same",	activation=tf.nn.tanh,name='conv1')
	conv1_2 = tf.layers.conv1d(inputs=conv1_1,filters=64,kernel_size=11,
			kernel_regularizer = regularizer,
			bias_regularizer = regularizer,
			padding="same",	activation=tf.nn.tanh)
	pool1 = tf.layers.max_pooling1d(inputs=conv1_2, pool_size=4, strides=4)

#	dropout1 = tf.layers.dropout(inputs=pool1, rate=drop)

#	x_mean, x_var = tf.nn.moments(dropout1, 0)
#	bn1 = tf.nn.batch_normalization(x=dropout1, mean=x_mean, variance=x_var, offset=None, scale=None, variance_epsilon=epsilon)
#	bn1 = tf.cond(is_train,
#					lambda: tf.contrib.layers.batch_norm(dropout1, is_training=True),
#					lambda: tf.contrib.layers.batch_norm(dropout1, is_training=False))
	bn1 = batch_norm(pool1, is_train, scope='bn1')

	conv2_1 = tf.layers.conv1d(inputs=bn1,filters=64,kernel_size=13,padding="same",activation=tf.nn.tanh,name='conv2')
	conv2_2 = tf.layers.conv1d(inputs=conv2_1,filters=64,kernel_size=13,padding="same",activation=tf.nn.tanh)
	pool2_1 = tf.layers.max_pooling1d(inputs=conv2_2, pool_size=2, strides=2)

	conv2_3 = tf.layers.conv1d(inputs=pool2_1,filters=50,kernel_size=13,padding="same",activation=tf.nn.tanh)
	pool2 = tf.layers.max_pooling1d(inputs=conv2_3, pool_size=2, strides=2)
	
	dropout2 = tf.layers.dropout(inputs=pool2, rate=drop)

	conv3_1 = tf.layers.conv1d(inputs=dropout2,filters=50,kernel_size=13,padding="same",activation=tf.nn.tanh,name='conv3_1')
	conv3_2 = tf.layers.conv1d(inputs=conv3_1,filters=50,kernel_size=13,padding="same",activation=tf.nn.tanh,name='conv3_2')
	pool3 = tf.layers.max_pooling1d(inputs=conv3_2, pool_size=4, strides=4)

	dropout3 = tf.layers.dropout(inputs=pool3, rate=drop)
	bn3 = batch_norm(dropout3, is_train, scope='bn3')
	
	conv4_1 = tf.layers.conv1d(inputs=bn3,filters=30,kernel_size=11,padding="same",activation=tf.nn.tanh)
	conv4_2 = tf.layers.conv1d(inputs=conv4_1,filters=30,kernel_size=11,padding="same",activation=tf.nn.tanh)
	pool4 = tf.layers.max_pooling1d(inputs=conv4_2, pool_size=4, strides=4)
	
	dropout4 = tf.layers.dropout(inputs=pool4, rate=drop)

#	conv5_1 = tf.layers.conv1d(inputs=dropout0,filters=50,kernel_size=11,padding="same",activation=tf.nn.tanh)
#	conv5_2 = tf.layers.conv1d(inputs=conv5_1,filters=50,kernel_size=11,padding="same",activation=tf.nn.tanh)
#	pool5 = tf.layers.max_pooling1d(inputs=conv5_2, pool_size=4, strides=4)
	
	pool_flat = tf.reshape(dropout4, [-1, int((n_input/256) * 30) ])

	dense1 = tf.layers.dense(inputs=pool_flat, units=30, activation=tf.nn.tanh)
#	dropout1 = tf.layers.dropout(inputs=dense0, rate=drop)
#	dense = tf.layers.dense(inputs=dropout1, units=2, activation=tf.nn.tanh)

	dropout5 = tf.layers.dropout(inputs=dense1, rate=drop)
	
	dense2 = tf.layers.dense(inputs=dropout5, units=n_classes)
	logit = tf.nn.softmax(dense2)

	kernel1 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'conv1/kernel')[0]
	kernel2 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'conv2/kernel')[0]
	kernel3_1 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'conv2/kernel')[0]
	kernel3_2 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'conv2/kernel')[0]

	return logit,dense1,(kernel1,kernel2,kernel3_1,kernel3_2)



def get_train_inputs(filename):
	dat = pd.read_table(filename,sep=',',header=None,index_col=None)
	trX = np.array(dat)
	y = trX[:,trX.shape[1]-1]
	x = trX[:,0:(trX.shape[1]-1)]

	return x, y


trX,trY = get_train_inputs(AF_TRAINING)
test_x2,test_y2 = get_train_inputs(AF_TEST)
print trX.shape; print trY.shape

rg = range(trX.shape[0])
select = random.sample(rg,20)
select2 = [j for j in rg if j not in select]

test_x = trX[select,]
test_y = trY[select,]

trX = trX[select2,]
trY = trY[select2,]

n_classes = len(set(trY))
print "n_classes: %d" % n_classes


#with tf.device('/gpu:1'):
if 1>0:
		pred, dense,kernel = cnn_ectraction(x, keep_prob,n_input,train_flag)
		k1,k2,k31,k32 = kernel

		feature_obtained = tf.concat(dense, 1, name="features")
		IDclass = tf.arg_max(pred, 1, name='IDclass')

		onehot_labels = tf.one_hot(indices=tf.cast(y, tf.int32), depth=n_classes)
#		cost = tf.losses.sigmoid_cross_entropy(onehot_labels, logits=pred) + 0.01*(tf.nn.l2_loss(k1) + tf.nn.l2_loss(k2) + tf.nn.l2_loss(k31) + tf.nn.l2_loss(k32))
#		cost = tf.losses.softmax_cross_entropy(onehot_labels, logits=pred) + 0.3*(tf.nn.l2_loss(k1) + tf.nn.l2_loss(k2) + tf.nn.l2_loss(k31) + tf.nn.l2_loss(k32))
		cost = tf.losses.sigmoid_cross_entropy(onehot_labels, logits=pred) + 0.3*tf.nn.l2_loss(k2) + 0.3*tf.nn.l2_loss(k31) + 0.3*tf.nn.l2_loss(k32)  
		#optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate).minimize(cost)
		optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
		optimizer2 = tf.train.AdamOptimizer(learning_rate=learning_rate2).minimize(cost)

		correct_pred = tf.equal(tf.arg_max(pred, 1), tf.arg_max(onehot_labels, 1))
		accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name="accuracy")

		init = tf.global_variables_initializer()

saver=tf.train.Saver()
prefix = argv[3]
epochs = 50000
read_log = True
with tf.Session() as sess:
	sess.run(init)
	#train_writer = tf.summary.FileWriter('log/train',sess.graph)
	if read_log:                    
		with open("log2/checkpoint",'r') as f1:
				txt = f1.readline()
				point = txt.strip().replace('model_checkpoint_path: ','').replace("\"",'')
				print point
				saver.restore(sess,"log2/%s"%point)	

	val_acc2 = 0
	for step in range(epochs):
			x_collection, y_collection = read_data(trX,trY,200)
			
			for n in range(len(x_collection)):
				batch_x = x_collection[n]
				batch_y = y_collection[n]
				
				if val_acc2 > 0.42:
					_op = sess.run(optimizer2, feed_dict={x: batch_x, y: batch_y, keep_prob: 0.3, train_flag: True})
				else:
					_op = sess.run(optimizer, feed_dict={x: batch_x, y: batch_y, keep_prob: 0.3, train_flag: True})
				#_op,loss,acc,output = sess.run((optimizer,cost,accuracy,output2), feed_dict={x: batch_x, y: batch_y, keep_prob: dropout})

			loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x, y: batch_y, train_flag: False})
			val_acc,IDc1 = sess.run((accuracy,IDclass), feed_dict={x: test_x, y: test_y, keep_prob: 0, train_flag: False})
			val_acc2,IDc2 = sess.run((accuracy,IDclass), feed_dict={x: test_x2, y: test_y2, keep_prob: 0, train_flag: False})

			if val_acc2 > 0.3:
				checkpoint_filepath='log2/train-%d_acc%f.ckpt' % (step,val_acc2)
				saver.save(sess,checkpoint_filepath)
				
			print "Epoch %d/%d - loss: %s - acc: %s\tvalidation acc: %s\t%s" % (step,epochs,str(loss),str(acc),str(val_acc),str(val_acc2))
			if val_acc2 > 0.5:	
				output_graph_def = graph_util.convert_variables_to_constants(sess, sess.graph_def, 
														output_node_names=["inputx", "inputy", 'keep_prob',  'features','IDclass'])
				with tf.gfile.FastGFile('./%s.acc%f.pb' %(prefix,val_acc2), mode='wb') as f:
						f.write(output_graph_def.SerializeToString())
	
	output_graph_def = graph_util.convert_variables_to_constants(sess, sess.graph_def,
																 output_node_names=["inputx", "inputy", 'keep_prob',  'features','IDclass'])
	with tf.gfile.FastGFile('./%s.epoch%d.pb' % (prefix,epochs), mode='wb') as f:
		f.write(output_graph_def.SerializeToString())







