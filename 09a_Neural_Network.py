from sklearn import datasets
import numpy as np

X_data = np.array([[0.4,0.3],[0.6,0.8],[0.7,0.5],[0.9,0.2]])
Y_data = np.array([[1],[1],[1],[0]])

X = X_data.T
Y = Y_data.T

W = np.zeros((X.shape[0], 1))
b = 0

num_samples = float(X.shape[1])
for i in range(1000):
    Z = np.dot(W.T,X) + b
    pred_y = 1/(1 + np.exp(-Z))
    if(i%100 == 0):
      print("cost after %d epoch:"%i)
      print (-1/num_samples *np.sum(Y*np.log(pred_y) + (1-Y)*(np.log(1-pred_y))))
    dW = (np.dot(X,(pred_y-Y).T))/num_samples
    db = np.sum(pred_y-Y)/num_samples
    W = W - (0.1 * dW)
    b = b - (0.1 * db)

print (W,b)