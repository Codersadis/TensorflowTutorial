import tensorflow as tf
import numpy as np

# Flags for weather save the session
S = False

if S: # save the session
	# Save to files
	# remember pre define the folder
	W = tf.Variable([ [1,2,3], [3,4,5] ], dtype=tf.float32, name='Weights')
	b  = tf.Variable( [[1,2,3]], dtype=tf.float32, name='bias' )

	init = tf.initialize_all_variables()
	saver = tf.train.Saver()	
	with tf.Session() as sess:
		sess.run(init)
		save_path = saver.save(sess, 'my_model/save_net.ckpt') # checkpoint - ckpt
		print "Save to path:", save_path
else: # reload the params
	W_r  =  tf.Variable(np.arange(6).reshape((2,3)), dtype=tf.float32, name='Weights')	
	b_r   =  tf.Variable(np.arange(3).reshape((1,3)), dtype=tf.float32, name='bias')	
	saver = tf.train.Saver()
	with tf.Session() as sess:
		saver.restore(sess, 'my_model/save_net.ckpt')
		print "Weights:", sess.run(W_r)
		print "bias:", sess.run(b_r)


