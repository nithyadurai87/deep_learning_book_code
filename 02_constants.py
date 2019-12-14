import tensorflow as tf

print (tf.constant("hello world!"))
print (tf.constant(1))
print (tf.constant(1, tf.int16))
print (tf.constant(1.25))
print (tf.constant([1,2,3]))	
print (tf.constant([[1,2,3],[4,5,6]]))	
print (tf.constant([[[1,2,3],[4,5,6]],
                    [[7,8,9],[10,11,12]],
                    [[13,14,15],[16,17,18]]]))	
print (tf.constant([True, True, False]))
print (tf.zeros(10))	
print (tf.ones([10, 10]))