import tensorflow as tf

a = tf.get_variable("v3", [3,3])
b = tf.get_variable("v4", initializer=tf.constant([[3, 3],[4, 4]]))

x = tf.Variable(1)
y = tf.constant(2)
r = tf.assign(x,tf.multiply(x,y))

with tf.Session() as s:    
    #print(s.run(a))    
    s.run(tf.global_variables_initializer())
    print(s.run(a))
    print(s.run(b))
    for i in range(10):
        print (s.run(r))
