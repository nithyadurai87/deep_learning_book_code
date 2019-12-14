import numpy as np
import tensorflow as tf
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split 
import pandas as pd  

def normalize(data):
  col_max = np.max(data, axis = 0)
  col_min = np.min(data, axis = 0)
  return np.divide(data - col_min, col_max - col_min)  

(X_cancer, y_cancer) = load_breast_cancer(return_X_y = True)
X_train, X_test, Y_train, Y_test = train_test_split(X_cancer, y_cancer, random_state = 25)

X_train = X_train[:,:2]

import matplotlib.pyplot as plt
import matplotlib.colors
colors=['blue','green']
cmap = matplotlib.colors.ListedColormap(colors)
plt.figure()
plt.title('Non-linearly separable classes')
plt.scatter(X_train[:,0], X_train[:,1], c=Y_train, marker= 'o', s=50,cmap=cmap,alpha = 0.5 )
plt.show()

def forwardProp(X, parameters, drop_out = False):
    A = X
    L = len(parameters)//2                              
    for l in range(1,L):                                 
        A_prev = A
        A = tf.nn.relu(tf.add(tf.matmul(parameters['W' + str(l)], A_prev), parameters['b' + str(l)]))               
        if drop_out == True:                                  
            A = tf.nn.dropout(x = A, keep_prob = 0.8)                                     
    A = tf.add(tf.matmul(parameters['W' + str(L)], A), parameters['b' + str(L)])   
    return A

def deep_net(regularization = False, lambd = 0, drop_out = False, optimizer = False):
    tf.reset_default_graph()
    layer_dims = [2,25,25,1]                     
    X = tf.placeholder(dtype = tf.float64, shape = ([layer_dims[0],None]))
    Y = tf.placeholder(dtype = tf.float64, shape = ([1,None]))            

    tf.set_random_seed(1)
    parameters = {}
    for i in range(1,len(layer_dims)):
        parameters['W' + str(i)] = tf.get_variable("W"+ str(i), shape=[layer_dims[i], layer_dims[i-1]], initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float64)
        parameters['b' + str(i)] = tf.get_variable("b"+ str(i), initializer=tf.zeros([layer_dims[i],1],dtype=tf.float64))    
                                
    Z_final = forwardProp(X, parameters, drop_out)        
    
    cost = tf.nn.sigmoid_cross_entropy_with_logits(logits=Z_final,labels=Y)
    if optimizer == "momentum":
        train_net =  tf.train.MomentumOptimizer(0.01, momentum=0.9).minimize(cost) 
    elif optimizer == "rmsProp":
        train_net = tf.train.RMSPropOptimizer(0.01, decay=0.999, epsilon=1e-10).minimize(cost)                   
    elif optimizer == "adam":
        train_net = tf.train.AdamOptimizer(0.01, beta1 = 0.9, beta2 = 0.999).minimize(cost)    
    if regularization:
        reg_term = 0                               
        L = len(parameters)//2                     
        for l in range(1,L+1):
            reg_term += tf.nn.l2_loss(parameters['W'+ str(l)])   
        cost = cost + (lambd/2) * reg_term
    cost = tf.reduce_mean(cost)
    
    train_net = tf.train.GradientDescentOptimizer(0.01).minimize(cost)  
    init = tf.global_variables_initializer()
    costs = []
    with tf.Session() as sess:
        sess.run(init)
        for i in range(10000):
            _,c = sess.run([train_net, cost], feed_dict={X: normalize(X_train).T, Y: Y_train.reshape(1, len(Y_train))})
            if i % 100 == 0:
                costs.append(c)
            if i % 1000 == 0:
                print(c)
        plt.ylim(min(costs)+0.1 ,max(costs), 4, 0.01)
        plt.xlabel("epoches per 100")
        plt.ylabel("cost")
        plt.plot(costs)
        plt.show()
        params = sess.run(parameters)
    return params

def predict(X, parameters):
    with tf.Session() as sess:
        Z = forwardProp(X, parameters, drop_out= False)
        A = sess.run(tf.round(tf.sigmoid(Z)))
    return A

def plot_decision_boundary1( X, y, model):
    plt.clf()
    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
    y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1   
    colors=['blue','green']
    cmap = matplotlib.colors.ListedColormap(colors)   
    h = 0.01
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    A = model(np.c_[xx.ravel(), yy.ravel()])
    A = A.reshape(xx.shape)
    plt.contourf(xx, yy, A, cmap="spring")
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.scatter(X[0, :], X[1, :], c=y, s=8,cmap=cmap)
    plt.title("Decision Boundary for learning rate:")
    plt.show()

p = deep_net(regularization = True, lambd = 0.02)
plot_decision_boundary1(normalize(X_train).T,Y_train,lambda x: predict(x.T,p))

p = deep_net(drop_out = True)
p = deep_net(optimizer="momentum")
p = deep_net(optimizer="rmsProp")
p = deep_net(optimizer="adam")
