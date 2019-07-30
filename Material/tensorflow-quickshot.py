import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

N, D, H = 100, 10, 64

x = tf.placeholder(tf.float32, shape=(N,D))
y = tf.placeholder(tf.float32, shape=(N,D))

init = tf.contrib.layers.xavier_initializer()
h = tf.layers.dense(inputs=x, units=H,
	activation=tf.nn.relu, kernel_initializer=init)
y_pred = tf.layers.dense(inputs=h, units=D, kernel_initializer=init)

loss = tf.losses.mean_squared_error(y_pred, y)

learning_rate = 1e-5
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
updates = optimizer.minimize(loss)

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	values = {x: np.random.randn(N, D),
			y: np.random.randn(N, D)}
	mid_loss = list()
	for t in range(50):
		loss_val, _ = sess.run([loss, updates], feed_dict=values)
		mid_loss.append(loss_val)
	plt.plot(list(range(50)),mid_loss)
	plt.show()
