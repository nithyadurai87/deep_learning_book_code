from sklearn.model_selection import train_test_split 
from sklearn.datasets import load_breast_cancer
import tensorflow as tf
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def normalize(data):
  col_max = np.max(data, axis = 0)
  col_min = np.min(data, axis = 0)
  return np.divide(data - col_min, col_max - col_min)  
    
(X_cancer, y_cancer) = load_breast_cancer(return_X_y = True)

cmap = matplotlib.colors.ListedColormap(['black','yellow'])
plt.figure()
plt.title('Non-linearly separable classes')
plt.scatter(X_cancer[:,0], X_cancer[:,1], c=y_cancer, marker= 'o', s=50, cmap=cmap, alpha = 0.5 )
plt.show()
plt.savefig('fig1.png', bbox_inches='tight')

X_train, X_test, Y_train, Y_test = train_test_split(X_cancer, y_cancer, random_state = 25)

X_train = normalize(X_train).T
Y_train = Y_train.reshape(1, len(Y_train))

X_test = normalize(X_test).T
Y_test = Y_test.reshape(1, len(Y_test))

X = tf.placeholder(dtype = tf.float64, shape = ([X_train.shape[0],None]))
Y = tf.placeholder(dtype = tf.float64, shape = ([1,None]))
  
W1 = tf.Variable(initial_value=tf.random_normal([8,X_train.shape[0]], dtype = tf.float64) * 0.01)
b1 = tf.Variable(initial_value=tf.zeros([8,1], dtype=tf.float64))
W2 = tf.Variable(initial_value=tf.random_normal([1,8], dtype=tf.float64) * 0.01)
b2 = tf.Variable(initial_value=tf.zeros([1,1], dtype=tf.float64))  

print (W1,b1,W2,b2)
  
Z1 = tf.matmul(W1,X) + b1
A1 = tf.nn.relu(Z1)
Z2 = tf.matmul(W2,A1) + b2
 
cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Z2,labels=Y))

GD = tf.train.GradientDescentOptimizer(0.2).minimize(cost)
  
with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  for i in range(5000):
    c = sess.run([GD, cost], feed_dict={X: X_train, Y: Y_train})[1]
    if i % 1000 == 0:
      print ("cost after %d epoch:"%i)
      print(c)        
  correct_prediction = tf.equal(tf.round(tf.sigmoid(Z2)), Y)
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
  print("Accuracy for training data:", accuracy.eval({X: X_train, Y: Y_train}))
  print("Accuracy for test data:", accuracy.eval({X: X_test, Y: Y_test}))