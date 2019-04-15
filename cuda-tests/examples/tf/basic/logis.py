#!/usr/bin/env python3

import tensorflow as tf

# dataMat = tf.constant([1.0, 1.0, 1.0, 1.0, 2.0, 2.0], shape=[2, 3])

with tf.device('/gpu:0'):
	dataMat = tf.placeholder(tf.float32, shape=[2, 3])
	labelMat = tf.constant([1.0, 2.0], shape=[2, 1])
	weight = tf.constant([1.0, 1.0, 1.0], shape=[3, 1])

	alpha = tf.constant(0.001)
	for i in range(500):
		h = tf.sigmoid(tf.matmul(dataMat, weight))
		error = tf.subtract(labelMat, h)
		weight = weight + alpha * tf.matmul(dataMat, error, transpose_a=True)

sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

print(sess.run(weight, feed_dict={dataMat: [[1, 2, 3], [4, 5, 6]]}))
