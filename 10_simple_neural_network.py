from sklearn import datasets
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split 

def normalize(data):
  col_max = np.max(data, axis = 0)
  col_min = np.min(data, axis = 0)
  return np.divide(data - col_min, col_max - col_min)  

def model(X,Y):
  num_samples = float(X.shape[1])
  W = np.zeros((X.shape[0], 1))
  b = 0
  for i in range(4000):
    Z = np.dot(W.T,X) + b
    A = 1/(1 + np.exp(-Z))
    if(i%100 == 0):
      print("cost after %d epoch:"%i)
      print (-1/num_samples *np.sum(Y*np.log(A) + (1-Y)*(np.log(1-A))))
    dW = (np.dot(X,(A-Y).T))/num_samples
    db = np.sum(A-Y)/num_samples
    W = W - (0.75 * dW)
    b = b - (0.75 * db)
  return W, b

def predict(X,W,b):
    Z = np.dot(W.T,X) + b
    i = []
    for y in 1/(1 + np.exp(-Z[0])):
        if y > 0.5 :
            i.append(1)
        else:
            i.append(0)  
    return (np.array(i).reshape(1,len(Z[0])))

(X_cancer, y_cancer) = load_breast_cancer(return_X_y = True)
X_train, X_test, y_train, y_test = train_test_split(X_cancer, y_cancer, random_state = 25)

X_train = normalize(X_train).T
y_train = y_train.T

X_test = normalize(X_test).T
y_test = y_test.T

Weights, bias = model(X_train, y_train)

y_predictions = predict(X_train,Weights,bias)
accuracy = 100 - np.mean(np.abs(y_predictions - y_train)) * 100
print ("Accuracy for training data: {} %".format(accuracy))
      
y_predictions = predict(X_test,Weights,bias)
accuracy = 100 - np.mean(np.abs(y_predictions - y_test)) * 100
print ("Accuracy for test data: {} %".format(accuracy))

