import tensorflow as tf

a = tf.constant("hello world!")
print (a)

s = tf.Session()
print (s.run(a))	
s.close()

with tf.Session() as s:    
    print(s.run(a)) 	