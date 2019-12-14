import numpy as np
import tensorflow as tf
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split 

def normalize(data):
  col_max = np.max(data, axis = 0)
  col_min = np.min(data, axis = 0)
  return np.divide(data - col_min, col_max - col_min)  

(X_cancer, y_cancer) = load_breast_cancer(return_X_y = True)

X_train, X_test, Y_train, Y_test = train_test_split(X_cancer, y_cancer, random_state = 25)

X_train = normalize(X_train).T
Y_train = Y_train.reshape(1, len(Y_train))

X_test = normalize(X_test).T
Y_test = Y_test.reshape(1, len(Y_test))

X = tf.placeholder(dtype = tf.float64, shape = ([X_train.shape[0],None]))
Y = tf.placeholder(dtype = tf.float64, shape = ([1,None]))

layer_dims = [X_train.shape[0],8,8,1]
parameters = {}
for i in range(1,len(layer_dims)):
  parameters['W' + str(i)] = tf.Variable(initial_value=tf.random_normal([layer_dims[i], layer_dims[i-1]], dtype=tf.float64)* 0.01)
  parameters['b' + str(i)] = tf.Variable(initial_value=tf.zeros([layer_dims[i],1],dtype=tf.float64) * 0.01)
  
A = X
L = int(len(parameters)/2)
for i in range(1,L):
  A_prev = A
  Z = tf.add(tf.matmul(parameters['W' + str(i)], A_prev), parameters['b' + str(i)])
  A = tf.nn.relu(Z) 
Z_final = tf.add(tf.matmul(parameters['W' + str(L)], A), parameters['b' + str(L)])

cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Z_final,labels=Y))

GD = tf.train.GradientDescentOptimizer(0.1).minimize(cost) 
  
with tf.Session() as sess:
  sess.run(tf.global_variables_initializer() )
  for i in range(5000):
    c = sess.run([GD, cost], feed_dict={X: X_train, Y: Y_train})[1]
    if i % 1000 == 0:
      print ("cost after %d epoch:"%i)
      print(c)
  correct_prediction = tf.equal(tf.round(tf.sigmoid(Z_final)), Y)
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
  print("Accuracy for training data:", accuracy.eval({X: X_train, Y: Y_train}))
  print("Accuracy for test data:", accuracy.eval({X: X_test, Y: Y_test}))
  
  
  

