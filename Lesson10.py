import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# load mnist data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# def function
def add_layer(inputs, in_size, out_size, activation_function=None,):
	# add one more layer and return the output of the layer
	Weights = tf.Variable(tf.random_normal([in_size, out_size]))
	bias = tf.Variable(tf.zeros([1, out_size])+0.1, )
	Wx_plus_b = tf.matmul(inputs, Weights) + bias
	if activation_function is None:
		outputs = Wx_plus_b
	else:
		outputs = activation_function(Wx_plus_b)
	return outputs

def compute_accuracy(v_xs, v_ys):
	global prediction
	y_pre = sess.run(prediction, feed_dict={xs:v_xs})
	correct_prediction = tf.equal( tf.argmax(y_pre, 1), tf.argmax(v_ys, 1) )
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	result  = sess.run(accuracy, feed_dict={xs:v_xs, ys:v_ys})
	return result


# define place holder
xs = tf.placeholder(tf.float32, [None, 784])
ys = tf.placeholder(tf.float32, [None, 10])

# add layers
prediction = add_layer(xs, 784, 10, activation_function=tf.nn.softmax)

# error
cross_entropy = tf.reduce_mean( -tf.reduce_sum(ys * tf.log(prediction), reduction_indices=[1]) )
train = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

with tf.Session() as sess:
	sess.run(tf.initialize_all_variables())

	for i in range(1000): 
		batch_xs, batch_ys = mnist.train.next_batch(100)  # SGD
		sess.run(train, feed_dict={xs:batch_xs, ys:batch_ys})
		if i % 50 == 0 :
			print(compute_accuracy( mnist.test.images, mnist.test.labels ))	



