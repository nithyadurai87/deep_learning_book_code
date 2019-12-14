import tensorflow as tf

x1 = tf.get_variable("a", dtype=tf.int32,  initializer=tf.constant([5]))
y1 = tf.get_variable("b", dtype=tf.int32,  initializer=tf.constant([6]))
c = tf.constant([5], name ="c")
f = tf.multiply(x1, y1) + tf.pow(x1, 2) + y1 + c	

with tf.Session() as s:  
    summary_writer = tf.summary.FileWriter('tensorboard_example',s.graph)
    s.run(tf.global_variables_initializer())
    print (s.run(f))
