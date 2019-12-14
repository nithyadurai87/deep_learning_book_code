import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def zero_pad(X, pad):
    X_padded = np.pad(array = X, pad_width = ((0,0),(pad,pad), (pad,pad),(0,0)), mode = 'constant', constant_values = 0) 
    return X_padded

def conv_single_step(X_slice, W, b):
    conv = np.multiply(X_slice, W)
    Z = np.sum(conv)
    Z = np.add(Z, b) 
    return Z

def conv_forward(X, W, b, hparams):
  stride = hparams["stride"]
  pad = hparams["pad"] 
  m, h_prev, w_prev, c_prev = X.shape   
  f, f, c_prev, n_c = W.shape
  
  n_h = int((h_prev - f + 2*pad)/stride) + 1
  n_w = int((w_prev - f + 2*pad)/stride) + 1
  
  Z = np.zeros((m, n_h, n_w, n_c))
  A_prev_pad = zero_pad(X, pad)
  for i in range(m):
    for h in range(n_h):
      for w in range(n_w):
        for c in range(n_c):
           w_start = w * stride
           w_end = w_start + f 
           h_start = h * stride
           h_end = h_start + f        
           Z[i,h,w,c] = conv_single_step(A_prev_pad[i, h_start:h_end, w_start:w_end, :], W[:,:,:,c], b[:,:,:,c])
  return Z

def max_pool(input, hparams):
    m, h_prev, w_prev, c_prev = input.shape
    f = hparams["f"]
    stride = hparams["stride"]
    h_out = int(((h_prev - f)/stride) + 1)
    w_out = int(((w_prev -f)/stride) + 1)
    output = np.zeros((m, h_out, w_out, c_prev))
    for i in range(m):
        for c in range(c_prev):
            for h in range(h_out):
                for w in range(w_out):
                    w_start = w * stride
                    w_end = w_start + f
                    h_start = h * stride
                    h_end = h_start + f
                    output[i, h, w, c] = np.max(input[i,h_start:h_end, w_start:w_end, c])
    assert output.shape == (m, h_out, w_out, c_prev)
    return output

img = mpimg.imread('./cake.JPG') 
print (img.shape) 
X = img.reshape(1,142,252,3) 

fig = plt.figure(figsize=(15,10))

ax1 = fig.add_subplot(2,2,1)
print("Shape of Image: ", X.shape)
ax1.imshow(X[0,:,:,:])
ax1.title.set_text('Original Image')

ax2 = fig.add_subplot(2,2,2)
X = zero_pad(X, 10)
print("After padding: ", X.shape)
ax2.imshow(X[0,:,:,:], cmap = "gray")
ax2.title.set_text('After padding')


ax3 = fig.add_subplot(2,2,3)
W = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]]).reshape((3,3,1,1))
b = np.zeros((1,1,1,1))
hparams = {"pad" : 0, "stride": 1}
X = conv_forward(X, W, b, hparams)
print("After convolution: ", X.shape)
ax3.imshow(X[0,:,:,0], cmap='gray',vmin=0, vmax=1)
ax3.title.set_text('After convolution')

ax4 = fig.add_subplot(2,2,4)
hparams = {"stride" : 1, "f" : 2}
X = max_pool(X, hparams)
print("After pooling :", X.shape)
ax4.imshow(X[0,:,:,0], cmap = "gray")
ax4.title.set_text('After pooling')

plt.show()
