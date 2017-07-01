import tensorflow as tf

def main():
	# Simple graph setup
	node1 = tf.constant(3.0, dtype=tf.float32)
	node2 = tf.constant(4.0) # Implicitly tf.float32

	# Print the nodes.  DOES NOT print out the values (i.e. 3.0 and 4.0 here)
	# We must run the graph in a session to evaluate the nodes
	print("%s %s" % (node1, node2))

	# Setup, run session
	# terminal: export TF_CPP_MIN_LOG_LEVEL=2 to silence warnings
	sess = tf.Session()
	print(sess.run([node1, node2]))

	# Add our nodes to produce a new graph
	node3 = tf.add(node1, node2)
	print("node3: %s" % node3)
	print("sess.run(node3): %s" % sess.run(node3))

	# Create a graph consisting of placeholders for the input, add
	a = tf.placeholder(tf.float32)
	b = tf.placeholder(tf.float32)
	adder_node = a + b

	# Use the graph with placeholders
	print("Adders:")
	print(sess.run(adder_node, {a: 3, b:4.5}))
	print(sess.run(adder_node, {a:[1,3], b: [2,4]}))

	# Add a multiplication operation
	print("Add and triple:")
	add_and_triple = adder_node * 3
	print(sess.run(add_and_triple, {a:3, b:4.5}))

	# Create graph with trainable parameters
	W = tf.Variable([.3], dtype=tf.float32)
	b = tf.Variable([-.3], dtype=tf.float32)
	x = tf.placeholder(tf.float32)
	linear_model = W * x + b

	# We must run the following to initialize our variables before running the session
	init = tf.global_variables_initializer()
	sess.run(init)

	# Evaluate our linear model for several values of x simultaneously
	print("linear model:")
	print(sess.run(linear_model, {x:[1,2,3,4]}))

	# Create a loss function and measure the loss of our model (standard model for linear regression)
	y = tf.placeholder(tf.float32)
	squared_deltas = tf.square(linear_model - y)
	loss = tf.reduce_sum(squared_deltas)
	print("Loss:")
	print(sess.run(loss, {x:[1,2,3,4], y:[0,-1,-2,-3]}))

	# We can of course improve our model manually:
	fixW = tf.assign(W, [-1.])
	fixb = tf.assign(b, [1.])
	sess.run([fixW, fixb])
	print("Loss after manual improvement:")
	print(sess.run(loss, {x:[1,2,3,4], y:[0,-1,-2,-3]}))


if __name__ == "__main__":
	main()