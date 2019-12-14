import tensorflow as tf

a = tf.constant(1) 
b = tf.constant(2)
c = tf.constant(1, name = "value") 
print (a.name)
print (b.name)
print (c.name)

d = tf.constant([[1,2],[3,4],[5,6]]) 
e = tf.ones(d.shape[0])
print (a.get_shape())
print (d.get_shape())
print (e.get_shape())	
#print (tf.ones(a.shape[0]))	

c = tf.constant(1.25)
print (c.dtype)
print (tf.cast(c, dtype=tf.int32).dtype)