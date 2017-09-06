#coding=utf-8
from __future__ import absolute_import
from __future__ import division
#from __future__ import print_function

#import tensorflow as tf
from resnet import *
import numpy as np
import pandas as pd
import random
from sys import argv
#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.4)

#sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

from tensorflow.python.framework import graph_util


AF_TRAINING = argv[1]
AF_TEST = argv[2]
val_filename = "/data/run/jack/heart_print/bayresianpart.csv"

n_input = 5120
n_classes = 221
learning_rate = 0.001
x = tf.placeholder(tf.float32, [None, n_input], name="inputx")
y = tf.placeholder(tf.int32, [None], name="inputy")
keep_prob = tf.placeholder(tf.float32, name="keep_prob")

def read_data(trX,trY,batch_size):
	rg_x = range(trX.shape[0])
	random.shuffle(rg_x)
	x_collection = [np.array(trX[rg_x[x:x+batch_size]]) for x in range(0,len(rg_x),batch_size)]
	y_collection = [np.array(trY[rg_x[x:x+batch_size]]) for x in range(0,len(rg_x),batch_size)]
	
	return x_collection,y_collection


def cnn_ectraction(x, drop,n_input,global_mean_var=None):
	epsilon = 0.001
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

	dropout1 = tf.layers.dropout(inputs=pool1, rate=drop)

	x_mean, x_var = tf.nn.moments(dropout1, [0,1])
	bn1 = tf.nn.batch_normalization(x=dropout1, mean=x_mean, variance=x_var, offset=None, scale=None, variance_epsilon=epsilon)

	conv2_1 = tf.layers.conv1d(inputs=bn1,filters=30,kernel_size=13,padding="same",activation=tf.nn.tanh,name='conv2')
	conv2_2 = tf.layers.conv1d(inputs=conv2_1,filters=30,kernel_size=13,padding="same",activation=tf.nn.tanh)
	pool2_1 = tf.layers.max_pooling1d(inputs=conv2_2, pool_size=4, strides=4)

	num_channel = pool2_1.get_shape().as_list()[-1]

	layers = [pool2_1]
	for ri in range(4):
		with tf.variable_scope('conv_res_1%d' % ri, reuse=False):
			res_conv = residual_block(layers[-1], num_channel)
			layers.append(res_conv)

	for ri in range(4):
		with tf.variable_scope('conv_res_2%d' % ri, reuse=False):
			res_conv = residual_block(layers[-1], num_channel*2)
			layers.append(res_conv)

	for ri in range(4):
		with tf.variable_scope('conv_res_3%d' % ri, reuse=False):
			res_conv = residual_block(layers[-1], num_channel*4)
			layers.append(res_conv)
	pool222 = layers[-1]
#	activation_summary(pool222)
#	assert conv3.get_shape().as_list()[1:] == [8, 8, 64]
	

#	conv2_3 = tf.layers.conv1d(inputs=pool222,filters=50,kernel_size=13,padding="same",activation=tf.nn.tanh)
#	pool2 = tf.layers.max_pooling1d(inputs=conv2_3, pool_size=2, strides=2)
	
	dropout2 = tf.layers.dropout(inputs=pool222, rate=drop)

	conv3_1 = tf.layers.conv1d(inputs=dropout2,filters=20,kernel_size=13,padding="same",activation=tf.nn.tanh,name='conv3_1')
	conv3_2 = tf.layers.conv1d(inputs=conv3_1,filters=20,kernel_size=13,padding="same",activation=tf.nn.tanh,name='conv3_2')
	pool3 = tf.layers.max_pooling1d(inputs=conv3_2, pool_size=4, strides=4)

	dropout3 = tf.layers.dropout(inputs=pool3, rate=drop)
	
	conv4_1 = tf.layers.conv1d(inputs=dropout3,filters=20,kernel_size=11,padding="same",activation=tf.nn.tanh)
	conv4_2 = tf.layers.conv1d(inputs=conv4_1,filters=20,kernel_size=11,padding="same",activation=tf.nn.tanh)
	pool4 = tf.layers.max_pooling1d(inputs=conv4_2, pool_size=4, strides=4)
	
	dropout4 = tf.layers.dropout(inputs=pool4, rate=drop)

#	conv5_1 = tf.layers.conv1d(inputs=dropout0,filters=50,kernel_size=11,padding="same",activation=tf.nn.tanh)
#	conv5_2 = tf.layers.conv1d(inputs=conv5_1,filters=50,kernel_size=11,padding="same",activation=tf.nn.tanh)
#	pool5 = tf.layers.max_pooling1d(inputs=conv5_2, pool_size=4, strides=4)
	
	pool_flat = tf.reshape(dropout3, [-1, int((n_input/256) * 20) ])

#	pool2_flat = tf.reshape(pool2, [-1, int((n_input/8))*100])


	dense1 = tf.layers.dense(inputs=pool_flat, units=50, activation=tf.nn.tanh)
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

# Load training and eval data
def get_train_inputs(filename):
	dat = pd.read_table(filename,sep=',',header=None,index_col=None)
	trX = np.array(dat)
	y = trX[:,trX.shape[1]-1]
	x = trX[:,0:(trX.shape[1]-1)]

	return x, y


pred, dense,kernel = cnn_ectraction(x, keep_prob,n_input)
k1,k2,k31,k32 = kernel

feature_obtained = tf.concat(dense, 1, name="features")
IDclass = tf.arg_max(pred, 1, name='IDclass')

onehot_labels = tf.one_hot(indices=tf.cast(y, tf.int32), depth=n_classes)
#cost = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=pred)
cost = tf.losses.sigmoid_cross_entropy(onehot_labels, logits=pred) + 0.2*(tf.nn.l2_loss(k2) + tf.nn.l2_loss(k31) + tf.nn.l2_loss(k32))
#optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate).minimize(cost)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

correct_pred = tf.equal(tf.arg_max(pred, 1), tf.arg_max(onehot_labels, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name="accuracy")
init = tf.global_variables_initializer()


epochs = 1000
with tf.Session() as sess:
	sess.run(init)
	#sess.run(local_init_op)
	step = 1
	
	trX,trY = get_train_inputs(AF_TRAINING)
	print trX.shape; print trY.shape

	rg = range(trX.shape[0])
	select = random.sample(rg,50)
	test_x = trX[select,]
	test_y = trY[select,]
	#print list(set([k for k in test_y]))
	select2 = [j for j in rg if j not in select]

	test_x2,test_y2 = get_train_inputs(AF_TEST)

	trX = trX[select2,]
	trY = trY[select2,]

#	val_x, val_y = get_train_inputs(val_filename)
	
	for step in range(epochs):
			x_collection, y_collection = read_data(trX,trY,500)
			
			for n in range(len(x_collection)):
		#	for n in range(1):
				batch_x = x_collection[n]
				batch_y = y_collection[n]
			#	print type(batch_y)
			#	print batch_y.shape
			#	print batch_y[0:3]			
	
				_op = sess.run(optimizer, feed_dict={x: batch_x, y: batch_y, keep_prob: 0.2})
				#_op,loss,acc,output = sess.run((optimizer,cost,accuracy,output2), feed_dict={x: batch_x, y: batch_y, keep_prob: dropout})

#			if step % 10 == 0:
			loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x, y: batch_y})
			val_acc,IDc1 = sess.run((accuracy,IDclass), feed_dict={x: test_x, y: test_y})
			val_acc2,IDc2 = sess.run((accuracy,IDclass), feed_dict={x: test_x2, y: test_y2})
				
			print "Epoch %d/%d - loss: %s - acc: %s\tvalidation acc: %s\t%s" % (step,epochs,str(loss),str(acc),str(val_acc),str(val_acc2))
#			print "Epoch %d/%d - loss: %s - acc: %s\tvalidation acc: %s" % (step,epochs,str(loss),str(acc),str(val_acc))
		
			if val_acc2 > 0.4:	
					output_graph_def = graph_util.convert_variables_to_constants(sess, sess.graph_def,output_node_names=["inputx", "inputy", 'keep_prob',  'features','IDclass'])
					with tf.gfile.FastGFile('./resNet.acc40.v1.pb', mode='wb') as f:
						f.write(output_graph_def.SerializeToString())
	print("Optimization Finished!")
	output_graph_def = graph_util.convert_variables_to_constants(sess, sess.graph_def,output_node_names=["inputx", "inputy", 'keep_prob',  'features','IDclass'])
	with tf.gfile.FastGFile('./resNet.1000.v1.pb', mode='wb') as f:
		f.write(output_graph_def.SerializeToString())


#	dens,acc = sess.run((dense,accuracy), feed_dict={x: val_x, y: val_y, keep_prob: 0.2})
#	dens,k11,k22 = sess.run((dense,k1,k2), feed_dict={x: val_x, y: val_y, keep_prob: 0.2})
#	dens_out = np.concatenate((dens,np.expand_dims(val_y,1)),axis=1)
	
#	np.savetxt('dense.csv',dens_out,fmt='%f')
#	print acc

"""
#training_file_path = "model/ecgdata2.mat"
#Y_path = "model/ecgatr2.mat"


dat = pd.read_table(training_file_path,index_col=0,sep=',',header=None)
trX = np.array(dat)[0:15000,]
trX2 = np.array(dat)[15000:30000,]
#trX = trX[:,1:trX.shape[1]]

dat = pd.read_table(Y_path,index_col=0,sep=',',header=None)
trY = np.array(dat)[0:15000,]
trY2 = np.array(dat)[15000:20000,]


x_zero_padding = np.zeros((trX.shape[0],512))
y_zero_padding = np.zeros((trX.shape[0],2))
trX = np.concatenate((trX,x_zero_padding),axis=1)
trY = np.concatenate((trY,y_zero_padding),axis=1)
trX2 = np.concatenate((trX2,x_zero_padding),axis=1)
trY2 = np.concatenate((trY2,y_zero_padding),axis=1)

n_input = trX.shape[1]
Y_len = trY.shape[1]
"""

"""
x = tf.placeholder(tf.float32, [None, n_input], name="inputx")
y = tf.placeholder(tf.int32, [None,Y_len], name="inputy")
keep_prob = tf.placeholder(tf.float32, name="dropout1")

output = cnn_net(x, keep_prob,n_input,Y_len)

#aFclass = tf.arg_max(logit, 1, name="aFclass")

onehot = tf.one_hot(indices=tf.cast(y, tf.int32), depth=2)
#cost = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=pred)
#cost = tf.losses.mean_squared_error(y,output)
#cost = tf.losses.sigmoid_cross_entropy(y,output)
#cost = tf.losses.sparse_softmax_cross_entropy(y,tf.squeeze(output))

# loss shape (1000, 127)  batch_size x label_length
cost = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,logits=output)

#fcn_loss(logits, labels, num_classes, head=None)
#cost,cost2 = fcn_loss(output,onehot_labels,4)
output2 = tf.arg_max(output,2)

#optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate).minimize(cost)
#tf.train.GradientDescentOptimizer
#tf.train.AdadeltaOptimizer
#tf.train.AdagradOptimizer


correct_pred = tf.equal(tf.cast(output2,tf.int32),y)
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name="accuracy")


# initialize variable
init = tf.global_variables_initializer()
local_init_op = tf.local_variables_initializer()

#			output = sess.run((output2), feed_dict={x: batch_x, y: batch_y, keep_prob: dropout})
#			np.savetxt('dense.csv',output,delimiter = ',')
			print output.shape

			print "Training Finished!"

"""
