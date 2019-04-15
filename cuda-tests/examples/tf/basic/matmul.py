#!/usr/bin/env python3

import tensorflow as tf
import random

x = tf.placeholder("float", [None, 3])
y = x * random.randint(3, 40)

with tf.Session() as session:
    x_data = [[1, 1, 3],
              [4, 5, 8],]
    result = session.run(y, feed_dict={x: x_data})
    print(">> result = ", result)
