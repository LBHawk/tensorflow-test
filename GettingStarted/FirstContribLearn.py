import numpy as np
import tensorflow as tf

# Suppress warnings
tf.logging.set_verbosity(tf.logging.ERROR)

# We create and train a simple linear regression using the higher level tf.contrib.learn api
def main():
	# Declare our list of features.  Here we only have one (real-valued) feature, x
	features = [tf.contrib.layers.real_valued_column("x", dimension=1)]

	# Create an estimator.  This is the front end to invoke training and evaluation.
	# Here, we make a linear regression estimator
	estimator = tf.contrib.learn.LinearRegressor(feature_columns=features)

	# Setup training and evaluation data sets
	x_train = np.array([1.,2.,3.,4.]) 			# Training input
	y_train = np.array([0.,-1.,-2.,-3.]) 		# Optimal output
	x_eval = np.array([2.,5.,8.,1.]) 			# Evaluation input
	y_eval = np.array([-1.01, -4.1, -7., .0]) 	# Evaluation optimal output

	# Our input function using our training data sets, a batch size,
	# and how many training epochs to do
	input_fn = tf.contrib.learn.io.numpy_input_fn({"x":x_train}, y_train,
													batch_size=4,
													num_epochs=1000)

	# Our evaluation function using our eval data sets
	eval_input_fn = tf.contrib.learn.io.numpy_input_fn(
		{"x":x_eval}, y_eval, batch_size=4, num_epochs=1000)

	# Invoke 1000 training steps on our model
	estimator.fit(input_fn=input_fn, steps=1000)

	# Evaluate our model using eval data sets
	train_loss = estimator.evaluate(input_fn=input_fn)
	eval_loss = estimator.evaluate(input_fn=eval_input_fn)
	print("train loss: %r" % train_loss)
	print("eval loss: %r" % eval_loss)



if __name__ == "__main__":
	main()