import tensorflow as tf
from sklearn.datasets import load_digits
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelBinarizer

# load data
digits = load_digits()
X = digits.data
y = digits.target
y = LabelBinarizer().fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3)

# def function
def add_layer(inputs, in_size, out_size, layer_name, activation_function=None,):
	# add one more layer and return the output of the layer
	Weights = tf.Variable(tf.random_normal([in_size, out_size]))
	bias = tf.Variable(tf.zeros([1, out_size])+0.1, )
	Wx_plus_b = tf.matmul(inputs, Weights) + bias
	Wx_plus_b = tf.nn.dropout(Wx_plus_b, keep_prob)

	if activation_function is None:
		outputs = Wx_plus_b
	else:
		outputs = activation_function(Wx_plus_b)
	tf.histogram_summary(layer_name + '/outputs', outputs)	
	return outputs

# place holder
keep_prob = tf.placeholder(tf.float32) # prob NOT to dropout
xs = tf.placeholder(tf.float32, [None, 64])
ys = tf.placeholder(tf.float32, [None, 10])

# add layer
l1 = add_layer(xs, 64, 50, 'l1', activation_function=tf.nn.tanh)
# l1 = tf.nn.relu(l1)
prediction = add_layer(l1, 50, 10, 'l2', activation_function=tf.nn.softmax)

# define loss
cross_entropy = tf.reduce_mean( -tf.reduce_sum(ys * tf.log(prediction), reduction_indices=[1]) )
tf.scalar_summary('loss', cross_entropy)
train = tf.train.GradientDescentOptimizer(0.6).minimize(cross_entropy)

with tf.Session() as sess:
	merged = tf.merge_all_summaries()
	train_writer = tf.train.SummaryWriter("logs/train", sess.graph)
	test_writer = tf.train.SummaryWriter("logs/test", sess.graph)	
	sess.run(tf.initialize_all_variables())

	for i in range(500):
		sess.run(train, feed_dict={xs:X_train, ys:y_train, keep_prob:0.6})
		if i % 50 == 0:
			# record loss
			train_result = sess.run(merged, feed_dict={xs:X_train, ys:y_train, keep_prob:1})
			test_result = sess.run(merged, feed_dict={xs:X_test, ys:y_test, keep_prob:1})
			train_writer.add_summary(train_result, i)
			test_writer.add_summary(test_result, i)


