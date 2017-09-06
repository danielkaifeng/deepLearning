from __future__ import absolute_import
from __future__ import division

#import tensorflow as tf
from resnet import *
import numpy as np
import pandas as pd
import random
from sys import argv
#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.4)
#sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
from tensorflow.python.framework import graph_util

n_input = 5120
learning_rate = 0.005
x = tf.placeholder(tf.float32, [None, n_input], name="inputx")
y = tf.placeholder(tf.int32, [None], name="inputy")
train_flag = tf.placeholder(tf.bool, name="is_train")
keep_prob = tf.placeholder(tf.float32, name="keep_prob")

def read_data(trX,trY,batch_size):
	rg_x = range(trX.shape[0])
	random.shuffle(rg_x)
	x_collection = [np.array(trX[rg_x[x:x+batch_size]]) for x in range(0,len(rg_x),batch_size)]
	y_collection = [np.array(trY[rg_x[x:x+batch_size]]) for x in range(0,len(rg_x),batch_size)]
	
	return x_collection,y_collection

# Load training and eval data
def get_train_inputs(filename):
	dat = pd.read_table(filename,sep=',',header=None,index_col=None)
	trX = np.array(dat)
	y = trX[:,trX.shape[1]-1]
	x = trX[:,0:(trX.shape[1]-1)]

	return x, y


trX,trY = get_train_inputs(argv[1])
test_x2,test_y2 = get_train_inputs(argv[2])
print trX.shape; print trY.shape

rg = range(trX.shape[0])
select = random.sample(rg,50)
test_x = trX[select,]
test_y = trY[select,]
select2 = [j for j in rg if j not in select]

trX = trX[select2,]
trY = trY[select2,]
n_classes = len(set(trY))

# model function from resnet module
pred, dense,kernel = cnn_ectraction(x, keep_prob,n_input,is_train=train_flag,n_classes=n_classes)
k1,k2,k31,k32 = kernel

feature_obtained = tf.concat(dense, 1, name="features")
IDclass = tf.arg_max(pred, 1, name='IDclass')

onehot_labels = tf.one_hot(indices=tf.cast(y, tf.int32), depth=n_classes)
cost = tf.losses.sigmoid_cross_entropy(onehot_labels, logits=pred) + 0.01*(tf.nn.l2_loss(k1) + tf.nn.l2_loss(k2) + tf.nn.l2_loss(k31) + tf.nn.l2_loss(k32))
#optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate).minimize(cost)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

correct_pred = tf.equal(tf.arg_max(pred, 1), tf.arg_max(onehot_labels, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name="accuracy")


epochs = 10
prefix = argv[3]
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())

	for step in range(epochs):
			x_collection, y_collection = read_data(trX,trY,200)
			
			for n in range(len(x_collection)):
				batch_x = x_collection[n]
				batch_y = y_collection[n]

				sess.run(optimizer, feed_dict={x: batch_x, y: batch_y,	train_flag: True, keep_prob: 0})
				#_op,loss,acc,output = sess.run((optimizer,cost,accuracy,output2), feed_dict={x: batch_x, y: batch_y, keep_prob: dropout})

#			if step % 10 == 0:
			loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x, y: batch_y})
			val_acc,IDc1 = sess.run((accuracy,IDclass), feed_dict={x: test_x, y: test_y, train_flag: False})
			val_acc2,IDc2 = sess.run((accuracy,IDclass), feed_dict={x: test_x2, y: test_y2, train_flag: False})
				
			print "Epoch %d/%d - loss: %s - acc: %s\tvalidation acc: %s\t%s" % (step,epochs,str(loss),str(acc),str(val_acc),str(val_acc2))
#			print "Epoch %d/%d - loss: %s - acc: %s\tvalidation acc: %s" % (step,epochs,str(loss),str(acc),str(val_acc))
		
			if val_acc2 > 0.01:	
					output_graph_def = graph_util.convert_variables_to_constants(sess, sess.graph_def,output_node_names=["inputx", "inputy", 'keep_prob',  'features','IDclass','is_train'])
					with tf.gfile.FastGFile('./%s.resNet.acc%f.pb' %(prefix,acc), mode='wb') as f:
						f.write(output_graph_def.SerializeToString())
					exit(0)

	output_graph_def = graph_util.convert_variables_to_constants(sess, sess.graph_def,
					output_node_names=["inputx", "inputy", 'keep_prob',  'features','IDclass','is_train'])

	with tf.gfile.FastGFile('./%s.resNet.epoch%d.pb' % (prefix,acc), mode='wb') as f:
		f.write(output_graph_def.SerializeToString())










