import tensorflow as tf

X_data = tf.constant([[0.4,0.3],[0.6,0.8],[0.7,0.5],[0.9,0.2]],name="input_value")
Y_data = tf.constant([[1.0],[1.0],[1.0],[0.0]],name="output_value")

X = tf.transpose(X_data)
Y = tf.transpose(Y_data)

W = tf.Variable(initial_value=tf.zeros([1,X.shape[0]], dtype=tf.float32),name="weight")
b = tf.Variable(initial_value=tf.zeros([1,1], dtype=tf.float32)) 

Z = tf.matmul(W,X) + b 
cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Z,labels=Y))
GD = tf.train.GradientDescentOptimizer(0.1).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(1000):
        c = sess.run([GD, cost])[1]
        if i % 100 == 0:
            print ("cost after %d epoch:"%i)
            print(c)
    print (sess.run(W),sess.run(b))




