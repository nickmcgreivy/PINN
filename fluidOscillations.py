import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.interpolate import griddata
import scipy
import scipy.io

import time

class fluidModelNN:
	
	def __init__(self,lb,ub,layers,x,xinitial,ninitial,uinitial,pinitial,xleft,xright):
		self.lb = lb
		self.ub = ub
		self.tinner = x[:,0]
		self.xinner = x[:,1]
		self.tinitial = xinitial[:,0]
		self.xinitial = xinitial[:,1]
		self.uinitial = uinitial
		self.ninitial = ninitial
		self.pinitial = pinitial
		self.tleft = xleft[:,0]
		self.xleft = xleft[:,1]
		self.tright = xright[:,0]
		self.xright = xright[:,1]
		self.layers = layers
		
		self.nweights, self.nbiases = self.initialize_NN(layers)
		self.uweights, self.ubiases = self.initialize_NN(layers)
		self.pweights, self.pbiases = self.initialize_NN(layers)        
		
		with tf.name_scope("Placeholders"):
			self.t = tf.placeholder(tf.float32, shape=[None])
			self.x = tf.placeholder(tf.float32, shape=[None])
			self.tl = tf.placeholder(tf.float32, shape=[None])
			self.tr = tf.placeholder(tf.float32, shape=[None])
			self.xl = tf.placeholder(tf.float32, shape=[None])
			self.xr = tf.placeholder(tf.float32, shape=[None])
			self.ti = tf.placeholder(tf.float32, shape=[None])
			self.xi = tf.placeholder(tf.float32, shape=[None])
			self.ui = tf.placeholder(tf.float32, shape=[None])
			self.ni = tf.placeholder(tf.float32, shape=[None])
			self.pi = tf.placeholder(tf.float32, shape=[None])
		
		self.n_net, self.u_net, self.p_net = self.net(self.t,self.x)
		self.ni_net, self.ui_net, self.pi_net = self.net(self.ti,self.xi)
		self.nl_net, self.ul_net, self.pl_net = self.net(self.tl,self.xl)
		self.nr_net, self.ur_net, self.pr_net = self.net(self.tr,self.xr)
		self.El = self.E(self.tl,self.xl)
		self.Er = self.E(self.tr,self.xr)
		self.Exl = self.E_x(self.tl,self.xl)
		self.Exr = self.E_x(self.tr,self.xr)
		
		
		self.continuityLoss = tf.reduce_mean(tf.square(self.continuity(self.t,self.x)))
		self.momentumLoss = tf.reduce_mean(tf.square(self.momentum(self.t,self.x)))
		self.gaussLoss = 2.*tf.reduce_mean(tf.square(self.gauss(self.t,self.x)))
		self.innerloss = self.continuityLoss + self.momentumLoss + self.gaussLoss
		
		self.niloss = 10.**tf.reduce_mean(tf.square(self.ni_net - self.ninitial))
		self.uiloss = 10.*tf.reduce_mean(tf.square(self.ui_net - self.uinitial))
		self.piloss = 10.*tf.reduce_mean(tf.square(self.pi_net - self.pinitial))
		self.initloss = self.niloss + self.uiloss + self.piloss
		
		self.nbloss =  tf.reduce_mean(tf.square(self.nl_net - self.nr_net))
		self.ubloss =  tf.reduce_mean(tf.square(self.ul_net - self.ur_net))
		self.pbloss = tf.reduce_mean(tf.square(self.pl_net - self.pr_net))
		self.Ebloss = tf.reduce_mean(tf.square(self.El - self.Er))
		self.Exbloss = tf.reduce_mean(tf.square(self.Exl - self.Exr))
		self.boundaryloss = self.nbloss + self.ubloss + self.pbloss + self.Ebloss + self.Exbloss
		
		self.loss = self.innerloss + self.initloss + .05*self.boundaryloss
		self.iters = 0
		
		
		self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss, 
																method = 'L-BFGS-B', 
																options = {'maxiter': 20000,
																		   'maxfun': 20000,
																		   'maxcor': 50,
																		   'maxls': 50,
																		   'ftol' : 1.0 * np.finfo(float).eps})
		tfconfig = tf.ConfigProto()
		tfconfig.gpu_options.allow_growth = True
		tfconfig.allow_soft_placement = True
		tfconfig.log_device_placement= False
		
		self.sess = tf.Session(config=tfconfig) 
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
		return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)
	
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
	
	def net(self, t,x):
		nnet = self.forward(tf.stack([t,x],axis=1),self.nweights,self.nbiases)
		unet = self.forward(tf.stack([t,x],axis=1),self.uweights,self.ubiases)
		pnet = self.forward(tf.stack([t,x],axis=1),self.pweights,self.pbiases)
		return nnet, unet, pnet # n, u, phi
	
	def continuity(self,t,x):
		n, u, _ = self.net(t,x)
		n_t = tf.gradients(n,t)[0]
		nu_x = tf.gradients(n*u,x)[0]
		return n_t + nu_x
	
	def momentum(self,t,x):
		_, u, phi = self.net(t,x)
		u_t = tf.gradients(u,t)[0]
		u_x = tf.gradients(u,x)[0]
		E = -tf.gradients(phi,x)[0]
		f = u_t + u * u_x + E
		return f
	
	def gauss(self,t,x):
		n, _, phi = self.net(t,x)
		E = -tf.gradients(phi,x)[0]
		E_x = tf.gradients(E,x)[0]
		f = E_x + n - 1.
		return f
	
	def E(self,t,x):
		_, _, phi = self.net(t,x)
		E = -tf.gradients(phi,x)[0]
		return E
	
	def E_x(self,t,x):
		E = self.E(t,x)
		E_x = tf.gradients(E,x)[0]
		return E_x
	
	def callbackBFGS(self,loss, continuityloss, momloss, gaussloss, initloss, \
					 boundaryloss, niloss, uiloss, piloss, nbloss, ubloss, \
					 pbloss, Ebloss, Exbloss):
		if self.iters % 50 == 0:
			print("Loss: {}".format(loss))
			print("Continuity loss: {} Momentum loss: {} Gauss loss: {}".format(continuityloss, momloss, gaussloss))
			print("Initial loss: {} Boundary loss: {}".format(initloss, boundaryloss))
			print("n initial loss: {} u initial loss: {} phi initial loss: {}".format(niloss, uiloss, piloss))
			print("nb loss: {} ub loss: {} phibloss: {}".format(nbloss, ubloss,pbloss))
			print("Eb loss: {} Exb loss: {}".format(Ebloss, Exbloss))
		self.iters += 1
		
	def close(self):
		self.sess.close()
	
	def trainBFGS(self):
		tf_dict = {self.t: self.tinner, self.x: self.xinner, self.tl: self.tleft, self.xl: self.xleft, self.tr: self.tright, \
				   self.xr: self.xright, self.ti: self.tinitial, self.xi:self.xinitial, self.ui:self.uinitial, \
				   self.ni: self.ninitial, self.pi: self.pinitial}
		self.optimizer.minimize(self.sess, 
								feed_dict = tf_dict,         
								fetches = [self.loss, self.continuityLoss, self.momentumLoss, self.gaussLoss, self.initloss,self.boundaryloss, \
										  self.niloss, self.uiloss, self.piloss, self.nbloss, self.ubloss, self.pbloss, self.Ebloss, \
										   self.Exbloss], 
								loss_callback = self.callbackBFGS) 
		
	def predict(self, X_star):
				
		n,u,p = self.sess.run(self.net(self.t,self.x), {self.t: X_star[:,0], self.x: X_star[:,1]})
		return n,u,p

def uInitialConditions(X, alpha):
    return (1 + alpha * np.sin(2*X))

def nInitialConditions(X, alpha):
    return 0.*X

def phiInitialConditions(X, alpha):
    return -(alpha * np.sin(2*X)/4.)

t_max = 1
alpha = 0.25
pi = math.pi
x_max = pi
N_x = int(math.sqrt(5000))
N_init = 200
N_boundary = 200
alpha = 0.25
layers = [2,50,50,50,50,50,50,50,1]

x = np.linspace(-x_max,x_max,N_x)
t = np.linspace(0,t_max,N_x)

T, X = np.meshgrid(t,x)

# Data inside the domain
x = np.hstack((T.flatten()[:,None], X.flatten()[:,None]))

# Domain bounds
lb = x.min(0)
ub = x.max(0)

# initial conditions
xinitial = np.hstack((np.zeros(N_init)[:,None],np.linspace(-x_max,x_max,N_init)[:,None]))
uinitial = uInitialConditions(xinitial[:,1],alpha)
ninitial = nInitialConditions(xinitial[:,1],alpha)
phiinitial = phiInitialConditions(xinitial[:,1],alpha)

t = np.linspace(0,t_max,N_boundary)

# boundary conditions
xleft = np.hstack((t[:,None],-x_max*np.ones(N_boundary)[:,None]))
xright = np.hstack((t[:,None],x_max*np.ones(N_boundary)[:,None]))

x = np.vstack((x,xinitial,xleft,xright))

model = fluidModelNN(lb,ub,layers,x,xinitial,ninitial,uinitial,phiinitial,xleft,xright)



model.trainBFGS()

n,u,phi = model.predict(x)
n_pred = griddata(x, n.flatten(), (T, X), method='cubic')
plt.imshow(n_pred.T, interpolation='nearest', cmap='rainbow', 
                  extent=[0, t_max, -x_max, x_max], 
                  origin='lower', aspect='auto')
plt.show()

import matplotlib.animation as animation
import numpy as np
from pylab import *


dpi = 100
fig, ax = plt.subplots()
x = np.linspace(-x_max,x_max,200)
T = np.ones(200) * t_max
t = 0.    
n, u, phi = model.predict(np.stack((t*T,x),axis=1))
linen, = ax.plot(n)
lineu, = ax.plot(u)
linephi, = ax.plot(phi)
def init():  # only required for blitting to give a clean slate.
    n, u, phi = model.predict(np.stack((t*T,x),axis=1))
    linen.set_ydata(n)
    lineu.set_ydata(u)
    linephi.set_ydata(phi)
    return linen, lineu, linephi

def animate(i):
    n,u,phi = model.predict(np.stack((i*T/10.,x),axis=1))
    linen.set_ydata(n)
    lineu.set_ydata(u)
    linephi.set_ydata(phi)
    return linen, lineu, linephi

ani = animation.FuncAnimation(fig, animate, init_func=init, interval=2, blit=True, save_count=10)
writer = animation.writers['ffmpeg'](fps=30)
ani.save('fluidWaves.mp4',writer=writer,dpi=dpi)
print("Done")
model.close()