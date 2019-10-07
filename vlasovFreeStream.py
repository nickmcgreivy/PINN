import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math
from pyDOE import lhs
from scipy.interpolate import griddata
import scipy
import scipy.io

import time

# Weights and biases
import wandb
wandb.init(project='pinn', entity="nick-and-phil")

np.random.seed(42)
tf.set_random_seed(42)

class vlasovFreeStreamNN:

    def __init__(self, lb, ub, layers, X_inner, X_i, u_i, X_b, X_t, generator):

        self.lb = lb
        self.ub = ub
        self.layers = layers
        self.generator = generator
        self.iters = 0

        # Initialize NNs
        self.weights, self.biases = self.initialize_NN(layers)

        with tf.name_scope("Inputs"):
            # f_inner tells us where to evaluate the function on the inside of the domain. It 
            # is a N by 2 tensor where the 0th index is t and the 1st index is x
            self.f_t = X_inner[:,0]
            self.f_x = X_inner[:,1]
            self.f_v = X_inner[:,2]
            self.X_inner = X_inner


            # initial conditions
            self.u_i_t = X_i[:,0]
            self.u_i_x = X_i[:,1]
            self.u_i_v = X_i[:,2]
            self.u_i = u_i
            self.X_i = X_i

            self.u_t_t = X_t[:,0]
            self.u_t_x = X_t[:,1]
            self.u_t_v = X_t[:,2]
            self.X_t = X_t

            self.u_b_t = X_b[:,0]
            self.u_b_x = X_b[:,1]
            self.u_b_v = X_b[:,2]
            self.X_b = X_b

        # Start a session
        tfconfig = tf.ConfigProto()
        tfconfig.gpu_options.allow_growth = True
        tfconfig.allow_soft_placement = True
        self.sess = tf.Session(config=tfconfig) 
                                        # log_device_placement=True))

        with tf.name_scope("Placeholders"):
            self.f__t   = tf.placeholder(tf.float32, shape=[None])
            self.f__x   = tf.placeholder(tf.float32, shape=[None])
            self.f__v   = tf.placeholder(tf.float32, shape=[None])
            self.u_i__t = tf.placeholder(tf.float32, shape=[None])
            self.u_i__x = tf.placeholder(tf.float32, shape=[None])
            self.u_i__v = tf.placeholder(tf.float32, shape=[None])
            self.u__i   = tf.placeholder(tf.float32, shape=[None])
            self.u_b__t = tf.placeholder(tf.float32, shape=[None])
            self.u_b__x = tf.placeholder(tf.float32, shape=[None])
            self.u_b__v = tf.placeholder(tf.float32, shape=[None])
            self.u_t__t = tf.placeholder(tf.float32, shape=[None])
            self.u_t__x = tf.placeholder(tf.float32, shape=[None])
            self.u_t__v = tf.placeholder(tf.float32, shape=[None])

        self.f_pred = self.net_f(self.f__t, self.f__x, self.f__v)
        self.u_pred = self.net_u(self.u_i__t, self.u_i__x, self.u_i__v)
        self.u_init_pred = self.net_u(self.u_i__t, self.u_i__x, self.u_i__v)
        self.u_b_pred = self.net_u(self.u_b__t,self.u_b__x,self.u_b__v)
        self.u_t_pred = self.net_u(self.u_t__t,self.u_t__x,self.u_t__v)
        self.u_db_pred = self.net_u_d(self.u_b__t,self.u_b__x,self.u_b__v)
        self.u_dt_pred = self.net_u_d(self.u_t__t,self.u_t__x,self.u_t__v)
        self.u_d_pred = self.net_u_d(self.u_i__t,self.u_i__x,self.u_i__v)

        self.fLoss = tf.reduce_mean(tf.square(self.f_pred))
        self.iLoss = tf.reduce_mean(tf.square(self.u_init_pred - self.u__i))
        self.bLoss = .5*tf.reduce_mean(tf.square(self.u_t_pred - self.u_b_pred)) +\
                     .5*tf.reduce_mean(tf.square(self.u_dt_pred - self.u_db_pred))
        self.fiLoss = self.iLoss + self.fLoss
        self.loss = self.fiLoss + self.bLoss
        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss, 
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

    def kaiming_init(self, size):
        in_dim = size[0]
        out_dim = size[1]        
        kaiming_stddev = np.sqrt(2/(in_dim))
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=kaiming_stddev), dtype=tf.float32)

    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]        
        xavier_stddev = np.sqrt(2/(in_dim + out_dim))
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)

    def forward(self, X, weights, biases):
        num_layers = len(weights) + 1
        H = 2.0*(X - self.lb)/(self.ub - self.lb) - 1.0
        for l in range(0,num_layers-2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y

    def net_u(self, t, x, v):
        u = self.forward(tf.stack([t,x,v],axis=1),self.weights,self.biases)
        return tf.squeeze(u)

    def net_u_d(self, t, x, v):
        u = self.net_u(t,x,v)
        u_x = tf.gradients(u, x)[0]
        return u_x

    def net_f(self, t, x, v):
        u = self.net_u(t,x,v)
        u_t = tf.gradients(u, t)[0]
        u_x = tf.gradients(u, x)[0]
        f = u_t + v * u_x
        return f

    def callback(self, loss, fLoss, iLoss, bLoss):
        self.iters += 1
        if self.iters % 20 == 0:
            print('Loss: {}'.format(loss))
            print('Interior loss: {} Initial loss: {} Boundary loss: {}'.format(fLoss, iLoss, bLoss))
            # Log loss on wandb
            wandb.log({'loss': loss, 'fLoss': fLoss, 'i:oss:': iLoss, 'bLoss': bLoss},
                      step=self.iters)

        #    u_pred, _, _ = self.predict(self.X_print)
        #    plt.imshow(np.reshape(u_pred,(100,100)))

    def train(self, numSteps, N_inner, N_i, N_b):
        for i in range(numSteps):
            for X_inner, X_i, u_i, X_b, X_t in self.generator(self.X_inner, self.X_i, self.u_i, self.X_b, \
                                                              self.X_t, N_inner, N_i, N_b):
                tf_dict = {self.f__t: X_inner[:,0], self.f__x: X_inner[:,1], self.f__v: X_inner[:,2], 
                         self.u_i__t: X_i[:,0], self.u_i__x: X_i[:,1], self.u_i__v: X_i[:,2],
                  self.u__i: u_i, self.u_b__t: X_b[:,0], self.u_b__x: X_b[:,1],
                  self.u_b__v: X_b[:,2], self.u_t__t: X_t[:,0], self.u_t__x: X_t[:,1],
                  self.u_t__v: X_t[:,2]}
                self.optimizer.minimize(self.sess, 
                                feed_dict = tf_dict,         
                                fetches = [self.loss, self.fLoss, self.iLoss, self.bLoss], 
                                loss_callback = self.callback)
            print(i)

    def save(self):
        tf.train.Saver().save(self.sess, "model.ckpt")

    def predict(self, X_star):
        u_p = self.sess.run(self.u_pred, {self.u_i__t: X_star[:,0], self.u_i__x: X_star[:,1], self.u_i__v: X_star[:,2]})
        d_p = self.sess.run(self.u_d_pred, {self.u_i__t: X_star[:,0], self.u_i__x: X_star[:,1], self.u_i__v: X_star[:,2]})
        f_p = self.sess.run(self.f_pred, {self.f__t: X_star[:,0], self.f__x: X_star[:,1], self.f__v: X_star[:,2]})
        return u_p, d_p, f_p

def initialConditions(X, V, alpha):
    return np.exp(-(V**2)) * (1 + alpha * np.sin(2*X))


def generator(X_inner, X_i, u_i, X_b, X_t, N_inner, N_i, N_b): 
    def sampleRows(X,N):
        idx = np.random.choice(np.shape(X)[0], N,replace=False)
        return X[idx,:]
    X_inner = sampleRows(X_inner,N_inner)
    I = sampleRows(np.stack([X_i[:,0],X_i[:,1],X_i[:,2],u_i],axis=1), N_i)
    X_i = I[:,0:3]
    u_i = I[:,3]
    X = sampleRows(np.stack([X_b[:,0],X_b[:,1],X_b[:,2],X_t[:,0],X_t[:,1],X_t[:,2]],axis=1), N_b)
    X_b = X[:,0:3]
    X_t = X[:,3:]
    X_inner = np.vstack([X_inner, X_i, X_b, X_t])
    yield (X_inner, X_i, u_i, X_b, X_t)

v_max = 3
t_max = 2*math.pi
alpha = 0.25
pi = math.pi
x_max = pi
N_f = 1000000
N_init = 300
N_pred = 300 # must be larger than N_u
layers = [3,20,20,20,20,20,20,20,1]

x = np.linspace(-x_max,x_max,N_pred)
t = np.linspace(0,t_max,N_pred)
v = np.linspace(-v_max,v_max,N_pred)

T, X, V = np.meshgrid(t,x,v)
X_star = np.hstack((T.flatten()[:,None], X.flatten()[:,None], V.flatten()[:,None]))

# Domain bounds
lb = X_star.min(0)
ub = X_star.max(0)

# boundary conditions
xx2 = np.stack((T[0,:,:], X[0,:,:], V[0,:,:]),axis=2) #x=-pi condition
xx2 = np.reshape(xx2,(N_pred**2,3))
xx3 = np.stack((T[-1,:,:], X[-1,:,:], V[-1,:,:]),axis=2) #x=pi condition
xx3 = np.reshape(xx3,(N_pred**2,3))

X_inner = lb[1:3] + (ub[1:3]-lb[1:3])*lhs(2,N_pred**2)

xx1 = np.stack((np.zeros(N_init**2), np.linspace(-x_max,x_max,N_init**2), np.clip(np.squeeze(np.random.multivariate_normal([0],[[1.5]],N_init**2)),-3,3)),axis=1)
uu1 = initialConditions(xx1[:,1],xx1[:,2],alpha) # exp(-v^2) (1 + alpha sin(2x))
X_u_train = np.vstack([xx1, xx2, xx3])

X_f_train = lb + (ub-lb)* lhs(3, N_f)

X_f_train = np.vstack((X_f_train, X_u_train))

model = vlasovFreeStreamNN(lb, ub, layers, X_f_train, xx1, uu1, xx2, xx3, generator)

with tf.device('/device:GPU:0'):
    start_time = time.time()
    model.train(2, 10000, 1000, 1000)
    elapsed = time.time() - start_time
    print('Training time: %.4f' % (elapsed))

t = np.linspace(0,t_max,100)
x = np.linspace(-pi,pi,100)
v = np.linspace(-v_max,v_max,100)

X, V = np.meshgrid(x,v)

Xr = np.reshape(X,(10000))
Vr = np.reshape(V,(10000))

print(np.shape(X))
u_initial_pred1, _, _ = model.predict(np.stack((np.zeros(10000),Xr,Vr),axis=1))
u_initial_pred1 = np.reshape(u_initial_pred1, (100,100))

plt.contour(x,v,u_initial_pred1)
plt.show()
plt.imshow(np.reshape(u_initial_pred1,(100,100)))
plt.show()
plt.plot(x,u_initial_pred1[50,:])
plt.show()
plt.plot(v,u_initial_pred1[:,50])
plt.show()

u_initial_pred2, d_pred,f_pred = model.predict(np.stack((np.ones(10000),Xr,Vr),axis=1))
print(np.shape(f_pred))
print(np.shape(d_pred))
u_initial_pred2 = np.reshape(u_initial_pred2, (100,100))

print(np.shape(d_pred))
print(np.shape(f_pred))

plt.pcolor(x,v,u_initial_pred2 - u_initial_pred1)
plt.show()

plt.imshow(u_initial_pred2)
plt.show()
print(model.predict(np.stack((np.zeros(10),-pi*np.ones(10),np.linspace(-v_max,v_max,10)),axis=1)))
print(xx1[0:10,:])

import matplotlib.animation as animation
import numpy as np
from pylab import *

dpi = 100

def ani_frame(model):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_aspect('equal')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    x = np.linspace(-pi,pi,100)
    v = np.linspace(-v_max,v_max,100)

    X, V = np.meshgrid(x,v)

    T = t_max * np.ones(10000)
    t = 0.

    Xr = np.reshape(X,(10000))
    Vr = np.reshape(V,(10000))
    u_initial_pred, _, _ = model.predict(np.stack((t*T,Xr,Vr),axis=1))
    u_initial_pred = np.reshape(u_initial_pred, (100,100))

    im = ax.imshow(u_initial_pred,interpolation='nearest',origin='lower')

    fig.set_size_inches([8,4])

    tight_layout()

    def update_img(n):
        u_pred, _, _ = model.predict(np.stack((n*T/100. + 0.01,Xr,Vr),axis=1))
        u_pred = np.reshape(u_pred, (100,100))
        im.set_data(u_pred)
        return im

    #legend(loc=0)
    ani = animation.FuncAnimation(fig,update_img,100,interval=30)
    writer = animation.writers['ffmpeg'](fps=30)

    ani.save('vlasovWorking.mp4',writer=writer,dpi=dpi)
    return ani

ani_frame(model)
print("Done")
