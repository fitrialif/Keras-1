import theano
import theano.tensor as T
import theano.tensor.nnet as nnet
import numpy as np
from theano.ifelse import ifelse


import os
import sys
scriptpath = "C:/Udemy/Pro Data Science in Python/Matrix_CV_ML.py"
sys.path.append(scriptpath)
import Matrix_CV_ML as ML_im
datax = ML_im.Matrix_CV_ML("C:/Udemy/Pro Data Science in Python/sq_vs_tri/tri",30,39)
datax.build_ML_matrix()

#datax.global_matrix = np.array(([44,23],[56,15],[150,132],[144,198]))
#datax.labels        = np.array([0,0,1,1])

x = T.matrix()
y = T.vector()


def layer(x, w):
    b = np.full((1,24),1)
    new_x = T.concatenate([x, b])
    m = T.dot(w.T, new_x)
    m = nnet.sigmoid(m)
    return m

def layer2(x, w):
    b = np.full((1,24),1)
    new_x = T.concatenate([x, b])
    m = T.dot(new_x.T, w )
    m = nnet.sigmoid(m)
    return m.T


def grad_desc(cost, theta):
    alpha = 0.01 #learning rate
    return theta - (alpha * T.grad(cost, wrt=theta))


theta1 = theano.shared(np.full((1171,3),-0.002,   dtype=np.float64)) # randomly initialize
theta2 = theano.shared(np.full((4,1),-0.002, dtype=np.float64))

hid1 = layer(x, theta1)
hid2 = layer2(hid1, theta2)


xent = -y * T.log(hid2) - (1-y) * T.log(1-hid2)


fc = xent.mean()

cost = theano.function(inputs=[x, y], outputs=fc, updates=[
        (theta1, grad_desc(fc, theta1)),(theta2, grad_desc(fc, theta2))])
run_forward = theano.function(inputs=[x], outputs=hid2)

  
inputs = datax.global_matrix.astype(np.float32)
exp_y = datax.labels.astype(np.float32)
cur_cost = 0
for i in range(5000):
        cur_cost = cost(inputs.T, exp_y)
        if i % 100 == 0: #only print the cost every 500 epochs/iterations (to save space)
            print('Cost: %s' % (cur_cost,))


