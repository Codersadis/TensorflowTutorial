import tensorflow as tf
import numpy as np

# create data
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data * 0.1 + 0.3

# crate tensorflow structure 
Weights = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
bias = tf.Variable(tf.zeros([1]))

# predict
y = Weights * x_data + bias


loss = tf.reduce_mean(tf.square(y - y_data)) # L2 loss
learning_rate = 0.1
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train = optimizer.minimize(loss)

# init all variables
init = tf.initialize_all_variables()


# tensorflow session
sess = tf.Session()
sess.run(init) # vital importance

for step in range(201):
    	sess.run(train)	
    	if step % 20 == 0:
    		print( "Step:" , step,  "param:", sess.run(Weights), sess.run(bias))




