import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data 
from random import randint

mnist = input_data.read_data_sets("MNIST_data/data", one_hot=True)
print (mnist.train.images.shape)
print (mnist.train.labels.shape)
print (mnist.test.images.shape)
print (mnist.test.labels.shape)

X = tf.placeholder(dtype = tf.float32, shape = (None,784), name='input')
Y = tf.placeholder(dtype = tf.float32, shape = ([None,10]))

W = tf.Variable(initial_value=tf.zeros([784, 10]), dtype = tf.float32)
b = tf.Variable(initial_value=tf.zeros([10], dtype=tf.float32))

XX = tf.reshape(X, [-1, 784])
Z = tf.nn.softmax(tf.matmul(XX, W) + b, name="output")

cost = -tf.reduce_mean(Y * tf.log(Z)) * 1000.0

GD = tf.train.GradientDescentOptimizer(0.005).minimize(cost)
correct_prediction = tf.equal(tf.argmax(Z, 1),tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter('log_mnist_softmax',graph=tf.get_default_graph())
    for epoch in range(10):
        batch_count = int(mnist.train.num_examples/100)
        for i in range(batch_count):
            batch_x, batch_y = mnist.train.next_batch(100)
            c = sess.run([GD, cost], feed_dict={X: batch_x, Y: batch_y})[1]
        print ("Epoch: ", epoch)
    print ("Accuracy: ", accuracy.eval(feed_dict={X: mnist.test.images,Y: mnist.test.labels}))
    print ("done")
    
    num = randint(1, mnist.test.images.shape[1]) 
    img = mnist.test.images[num] 
 
    classification = sess.run(tf.argmax(Z, 1), feed_dict={X: [img]}) 
    print('Neural Network predicted', classification[0])
    print('Real label is:', np.argmax(mnist.test.labels[num])) 

