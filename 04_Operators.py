import tensorflow as tf

a = tf.constant(121.5)
b = tf.constant([[1,2],[3,4],[5,6]]) 
c = tf.constant([[7,8],[9,10],[11,12]]) 

r1 = tf.sqrt(a)
r2 = tf.add(b,c)
r3 = tf.multiply(b,c)

with tf.Session() as s:  
    print (s.run(r1))	
    print (s.run(r2)) 
    print (s.run(r3)) 


