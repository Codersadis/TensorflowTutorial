import tensorflow as tf

# define layer 
def add_layer(inputs, in_size, out_size, activation_function=None):
	Weights = tf.Variables(tf.random_normal([in_size, out_size]))
	bias = tf.Variables(tf.zeros([1, out_size])+ 0.1)  # recommend not as zeros when init
	Wx_plus_b = tf.matmul(inputs, Weights) + bias	
	if  activation_function is None:
		outputs = Wx_plus_b
	else:
		outputs = activation_function(Wx_plus_b)
	return outputs







