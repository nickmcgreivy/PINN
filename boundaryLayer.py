import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.interpolate import griddata
import scipy
import scipy.io

import time

class boundaryLayerNN:
    
    def __init__(self, lb, ub, xb, xt, epsilon, layers, x, generator, N_batch):
        
        self.lb = lb
        self.ub = ub
        self.epsilon = tf.Variable(epsilon,trainable=False)
        self.wf = tf.Variable(.05,trainable=False)
        self.wz = tf.Variable(15.,trainable=False)
        self.wo = tf.Variable(15.,trainable=False)
        self.u_b = xb.astype(np.float32)
        self.u_t = xt.astype(np.float32)
        self.x = x
        self.layers = layers
        self.generator = generator
        self.N_batch = N_batch
        
        self.weights, self.biases = self.initialize_NN(layers)
          
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, 
                                        log_device_placement=True))      
        self.f = tf.placeholder(tf.float32, shape=[None,1])
        self.f_pred = self.net_f(self.f)
        self.u_pred = self.net_u(self.f)
        self.u_b_pred = self.net_u(self.u_b)
        self.u_t_pred = self.net_u(self.u_t)
        

        self.lossf = self.wf*tf.reduce_mean(tf.square(self.f_pred))
        self.loss_zero = self.wz*tf.reduce_mean(tf.square(self.u_b_pred - 1.))
        self.loss_one =  self.wo*tf.reduce_mean(tf.square(self.u_t_pred - 2.))
        self.loss = self.lossf + self.loss_zero + self.loss_one
        self.optimizer = tf.train.AdamOptimizer(0.001,epsilon=1e-5).minimize(self.loss)
        self.optimizer2 = tf.contrib.opt.ScipyOptimizerInterface(self.loss, 
                                                                method = 'L-BFGS-B', 
                                                                options = {'maxiter': 40000,
                                                                           'maxfun': 100000,
                                                                           'maxcor': 50,
                                                                           'maxls': 50,
                                                                           'ftol' : 1.0 * np.finfo(float).eps})
        
                
        init = tf.global_variables_initializer()
        self.sess.run(init)
        
    
    def initialize_NN(self, layers):        
        weights = []
        biases = []
        num_layers = len(layers) 
        for l in range(0,num_layers-1):
            W = self.xavier_init(size=[layers[l], layers[l+1]])
            b = tf.Variable(tf.zeros([1,layers[l+1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)        
        return weights, biases
        
    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]        
        xavier_stddev = np.sqrt(2/(in_dim + out_dim))
        return tf.Variable(tf.random.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)
    
    def forward(self, X, weights, biases):
        num_layers = len(weights) + 1
        
        H = 2.0*(X - self.lb)/(self.ub - self.lb) - 1.0
        for l in range(0,num_layers-2):
            W = weights[l]
            b = biases[l]
            H = tf.nn.tanh(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y
    
    def net_u(self, x):
        u = self.forward(x,self.weights,self.biases)
        return tf.squeeze(u)
    
    def net_f(self, x):
        u = self.net_u(x)
        u_x = tf.squeeze(tf.gradients(u, x))
        u_xx = tf.squeeze(tf.gradients(u_x, x))
        f = self.epsilon * u_xx + tf.squeeze((1+x))*u_x + u
        return f
    
    def setEpsilon(self,epsilon):
        op = self.epsilon.assign(epsilon)
        self.sess.run(op)

    def setWeights(self,wf,wz,wo):
        op = self.wf.assign(wf)
        self.sess.run(op)    
        op = self.wz.assign(wz)
        self.sess.run(op)  
        op = self.wo.assign(wo)
        self.sess.run(op)  
        
    def callbackSGD(self, loss,lf, lz, lo, i):
        if (i % 100 == 0):
            print(i)
            print("fLoss: {} zero loss: {} one loss: {}".format(lf, lz, lo))
            print('Loss:', loss)
    
    def callbackBFGS(self, loss,lf, lz, lo):
        print("fLoss: {} zero loss: {} one loss: {}".format(lf, lz, lo))
        print('Loss:', loss)
    
    def trainSGD(self, numSteps):
        for i in range(numSteps):
            for X in self.generator(self.x, self.N_batch):
                tf_dict = {self.f: X}
                _, l, lf, lz, lo  = self.sess.run([self.optimizer, self.loss, self.lossf, self.loss_zero, self.loss_one], feed_dict=tf_dict)
                self.callbackSGD(l, lf, lz, lo, i)
                
    def trainBFGS(self):
        tf_dict = {self.f: self.x}
        self.optimizer2.minimize(self.sess, 
                                feed_dict = tf_dict,         
                                fetches = [self.loss,self.lossf, self.loss_zero, self.loss_one], 
                                loss_callback = self.callbackBFGS)            

                
    def save(self):
        saver.save(self.sess, "model.ckpt")
    
    def predict(self, X_star):
        u_pred =  self.sess.run(self.u_pred, {self.f: X_star})
        f_pred = self.sess.run(self.f_pred, {self.f: X_star})
        return u_pred, f_pred

    def closeSess(self):
    	self.sess.close()
        
epsilon = 0.01

N = 100000
N_batch = 100
x = np.reshape(np.linspace(0,1,N),(N,1))
layers = [1,20,20,20,20,20,1]
b = np.reshape(np.asarray([0.]),(1,1))
t = np.reshape(np.asarray([1.]),(1,1))

def generator(x,N_batch):
    yield (x[np.random.randint(np.shape(x)[0], size=N_batch),:])

model = boundaryLayerNN(0.,1.,b,t,epsilon,layers,x,generator,N_batch)


model.trainSGD(1000)
model.trainBFGS()


xStar = np.reshape(np.linspace(0,1,1000),(1000,1))

u_pred, f = model.predict(xStar)
print(np.shape(f))
plt.plot(u_pred)
plt.savefig('epsilon01.png')

model.setEpsilon(0.005)
model.trainBFGS()


xStar = np.reshape(np.linspace(0,1,1000),(1000,1))

u_pred, f = model.predict(xStar)
print(np.shape(f))
plt.plot(u_pred)
plt.savefig('epsilon005.png')
