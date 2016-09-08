import tensorflow as tf
import numpy as np 

# define layer 
def add_layer(inputs, in_size, out_size, n_layer, activation_function=None):
	with tf.name_scope('layer'):
		layer_name = 'layer%s' % n_layer
		with tf.name_scope('weights'):
			Weights = tf.Variable(tf.random_normal([in_size, out_size]), name='Weighrs')
			tf.histogram_summary(layer_name+'/Weights', Weights)
		with tf.name_scope('bias'):			
			bias = tf.Variable(tf.zeros([1, out_size])+ 0.1, name='bias')  # recommend not as zeros when init
			tf.histogram_summary(layer_name+'/bias', bias)
		with tf.name_scope('Wx_plus_b'):
			Wx_plus_b = tf.matmul(inputs, Weights) + bias	
		if  activation_function is None:
			outputs = Wx_plus_b
		else:
			outputs = activation_function(Wx_plus_b)
		tf.histogram_summary(layer_name+'/outputs', outputs)			
		return outputs

# data definition
x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.square(x_data) - 0.5	+ noise

with tf.name_scope('inputs'): 
	xs = tf.placeholder(tf.float32, [None, 1], name='x_input')
	ys = tf.placeholder(tf.float32, [None, 1], name='y_input')

# network archietecture
l1 = add_layer(xs,  1, 10, n_layer=1, activation_function=tf.nn.relu)
prediction = add_layer(l1, 10, 1, n_layer=2, activation_function=None)

# training
# loss + regularization ->  train 
with tf.name_scope('loss'):
	loss =  tf.reduce_mean( tf.reduce_sum(tf.square(prediction - ys), 
				reduction_indices=[1]) )
	tf.scalar_summary('loss', loss)

with tf.name_scope('train'):
	learningrate = 0.1
	train_step = tf.train.GradientDescentOptimizer( learningrate ).minimize(loss) 

sess = tf.Session()
merged = tf.merge_all_summaries()
writer = tf.train.SummaryWriter("logs/", sess.graph)
sess.run(tf.initialize_all_variables())

for i in range(1000):
	sess.run( train_step, feed_dict={xs:x_data, ys:y_data} )
	if i % 50 == 0:
		result = sess.run(merged, 
			feed_dict={xs:x_data, ys:y_data})
	writer.add_summary(result, i)	


