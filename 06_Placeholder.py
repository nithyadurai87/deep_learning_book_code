import tensorflow as tf
import numpy as np

x = tf.placeholder(tf.float32,name="x")
y = tf.placeholder(tf.float32,[1],name="y")
z = tf.constant(2.0)
y = x * z
    
print (x)
with tf.Session() as s:
    print (s.run(y,feed_dict={x:[100]}))
    print (s.run(y,{x:[200]}))
    print (s.run(y,{x: np.random.rand(1, 10)}))
    print (s.run(tf.pow(x, 2),{x:[300]}))
