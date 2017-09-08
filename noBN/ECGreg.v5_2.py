#coding=utf-8
from __future__ import absolute_import
from __future__ import division

import tensorflow as tf
import numpy as np
import pandas as pd
import random
from sys import argv
#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
#sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

from tensorflow.python.framework import graph_util


AF_TRAINING = argv[1]
AF_TEST = argv[2]

n_input = 5120
learning_rate = 0.0002
x = tf.placeholder(tf.float32, [None, n_input], name="inputx")
y = tf.placeholder(tf.int32, [None], name="inputy")
keep_prob = tf.placeholder(tf.float32, name="keep_prob")

def read_data(trX,trY,batch_size):
	rg_x = range(trX.shape[0])
	random.shuffle(rg_x)
	x_collection = [np.array(trX[rg_x[x:x+batch_size]]) for x in range(0,len(rg_x),batch_size)]
	y_collection = [np.array(trY[rg_x[x:x+batch_size]]) for x in range(0,len(rg_x),batch_size)]
	
	return x_collection,y_collection


def cnn_ectraction(x, drop,n_input):
	input_layer = tf.reshape(x, [-1, n_input, 1])
	regularizer = tf.contrib.layers.l2_regularizer(scale=0.01)

	conv1_1 = tf.layers.conv1d(inputs=input_layer,filters=100,kernel_size=11,
				kernel_regularizer = regularizer,bias_regularizer = regularizer,
				padding="same",	activation=tf.nn.tanh,name='conv1')
	conv1_2 = tf.layers.conv1d(inputs=conv1_1,filters=100,kernel_size=11,
				kernel_regularizer = regularizer,bias_regularizer = regularizer,
				padding="same",	activation=tf.nn.tanh)
	pool1 = tf.layers.max_pooling1d(inputs=conv1_2, pool_size=4, strides=4)

	conv2_1 = tf.layers.conv1d(inputs=pool1,filters=100,kernel_size=15,padding="same",activation=tf.nn.tanh,name='conv2')
	conv2_2 = tf.layers.conv1d(inputs=conv2_1,filters=100,kernel_size=15,padding="same",activation=tf.nn.tanh)
	pool2 = tf.layers.max_pooling1d(inputs=conv2_2, pool_size=2, strides=2)
	
#	dropout2 = tf.layers.dropout(inputs=pool2, rate=drop)

	conv3_1 = tf.layers.conv1d(inputs=pool2,filters=50,kernel_size=8,padding="same",activation=tf.nn.tanh,name='conv3_1')
	conv3_2 = tf.layers.conv1d(inputs=conv3_1,filters=50,kernel_size=8,padding="same",activation=tf.nn.tanh,name='conv3_2')
	pool3 = tf.layers.max_pooling1d(inputs=conv3_2, pool_size=2, strides=2)

	dropout3 = tf.layers.dropout(inputs=pool3, rate=drop)
	
	conv4_1 = tf.layers.conv1d(inputs=dropout3,filters=30,kernel_size=11,padding="same",activation=tf.nn.tanh)
	conv4_2 = tf.layers.conv1d(inputs=conv4_1,filters=30,kernel_size=11,padding="same",activation=tf.nn.tanh)
	pool4 = tf.layers.max_pooling1d(inputs=conv4_2, pool_size=4, strides=4)
	
	dropout4 = tf.layers.dropout(inputs=pool4, rate=drop)

	conv5_1 = tf.layers.conv1d(inputs=dropout4,filters=20,kernel_size=11,padding="same",activation=tf.nn.tanh)
	conv5_2 = tf.layers.conv1d(inputs=conv5_1,filters=20,kernel_size=11,padding="same",activation=tf.nn.tanh)
	pool5 = tf.layers.max_pooling1d(inputs=conv5_2, pool_size=4, strides=4)
	
	pool_flat = tf.reshape(pool5, [-1, int((n_input/256) * 20) ])



	dense1 = tf.layers.dense(inputs=pool_flat, units=50, activation=tf.nn.tanh)

	dropout5 = tf.layers.dropout(inputs=dense1, rate=drop)
	
	dense2 = tf.layers.dense(inputs=dropout5, units=n_classes)
	logit = tf.nn.softmax(dense2)

	kernel1 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'conv1/kernel')[0]
	kernel2 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'conv2/kernel')[0]
	kernel3_1 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'conv2/kernel')[0]
	kernel3_2 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'conv2/kernel')[0]

	return logit,dense1,(kernel1,kernel2,kernel3_1,kernel3_2)


def get_train_inputs(filename):
	dat = pd.read_table(filename,sep=',',header=0,index_col=None)
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


with tf.device('/gpu:1'):
		pred, dense,kernel = cnn_ectraction(x, keep_prob,n_input)
		k1,k2,k31,k32 = kernel

		feature_obtained = tf.concat(dense, 1, name="features")
		IDclass = tf.arg_max(pred, 1, name='IDclass')

		onehot_labels = tf.one_hot(indices=tf.cast(y, tf.int32), depth=n_classes)
		cost = tf.losses.sigmoid_cross_entropy(onehot_labels, logits=pred) + 0.01*(tf.nn.l2_loss(k1) + tf.nn.l2_loss(k2) + tf.nn.l2_loss(k31) + tf.nn.l2_loss(k32))
		#optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate).minimize(cost)
		optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

		correct_pred = tf.equal(tf.arg_max(pred, 1), tf.arg_max(onehot_labels, 1))
		accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name="accuracy")

		init = tf.global_variables_initializer()

prefix = argv[3]
epochs = 1000
with tf.Session() as sess:
	sess.run(init)
	train_writer = tf.summary.FileWriter('log/train',sess.graph)
	
	for step in range(epochs):
			x_collection, y_collection = read_data(trX,trY,200)
			
			for n in range(len(x_collection))[0:3]:
				batch_x = x_collection[n]
				batch_y = y_collection[n]
	
				_op = sess.run(optimizer, feed_dict={x: batch_x, y: batch_y, keep_prob: 0.1})
				#_op,loss,acc,output = sess.run((optimizer,cost,accuracy,output2), feed_dict={x: batch_x, y: batch_y, keep_prob: dropout})

#			if step % 10 == 0:
			loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x, y: batch_y})
			val_acc,IDc1 = sess.run((accuracy,IDclass), feed_dict={x: test_x, y: test_y, keep_prob: 0})
			val_acc2,IDc2 = sess.run((accuracy,IDclass), feed_dict={x: test_x2, y: test_y2, keep_prob: 0})
				
			print "Epoch %d/%d - loss: %s - acc: %s\tvalidation acc: %s\t%s" % (step,epochs,str(loss),str(acc),str(val_acc),str(val_acc2))
			if val_acc2 > 0.4:	
				output_graph_def = graph_util.convert_variables_to_constants(sess, sess.graph_def, 
														output_node_names=["inputx", "inputy", 'keep_prob',  'features','IDclass'])
				with tf.gfile.FastGFile('./%s.resNet.acc%f.pb' %(prefix,val_acc2), mode='wb') as f:
						f.write(output_graph_def.SerializeToString())
			
	output_graph_def = graph_util.convert_variables_to_constants(sess, sess.graph_def,
																 output_node_names=["inputx", "inputy", 'keep_prob',  'features','IDclass'])
	with tf.gfile.FastGFile('./%s.noBN.epoch%d.pb' % (prefix,epochs), mode='wb') as f:
		f.write(output_graph_def.SerializeToString())







