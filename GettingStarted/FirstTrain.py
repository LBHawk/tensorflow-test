import numpy as np
import tensorflow as tf

def main():
	# Model Parameters
	W = tf.Variable([.3], dtype=tf.float32)
	b = tf.Variable([-.3], dtype=tf.float32)

	# Model input and output
	x = tf.placeholder(tf.float32)
	linear_model = W * x + b
	y = tf.placeholder(tf.float32)

	# Loss function
	loss = tf.reduce_sum(tf.square(linear_model - y))

	# Optimizer (training)
	optimizer = tf.train.GradientDescentOptimizer(0.01)
	train = optimizer.minimize(loss)

	# Training data
	x_train = [1,2,3,4] 	# Our inputs to train on
	y_train = [0,-1,-2,-3] 	# Our optimal output from feeding our input thru our model

	# Training loop
	init = tf.global_variables_initializer()
	sess = tf.Session()
	sess.run(init) 	# Initializes our model's variables (to 0.3, -0.3 here)
	for i in range(1000):
		sess.run(train, {x:x_train, y:y_train})

	# Evaluate training accuracy
	curr_W, curr_b, curr_loss = sess.run([W, b, loss], {x:x_train, y:y_train})
	print("W: %s, b: %s, loss: %s" % (curr_W, curr_b, curr_loss))



if __name__ == "__main__":
	main()