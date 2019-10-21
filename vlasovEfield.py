!pip install pyDOE
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math
from pyDOE import lhs
from scipy.interpolate import griddata
import scipy
import scipy.io

import time

np.random.seed(42)
tf.set_random_seed(42)

class vlasov1DNN:
	
	def __init__(self, lb, ub, layers, layersE, X_inner, E_inner, X_i, f_i, X_b, X_t, X_eta, E_b, E_t, X_E0, E_0, generator):
		
		self.lb = lb
		self.ub = ub
		self.lbE = lb[0:2]
		self.ubE = ub[0:2]
		self.layers = layers
		self.layersE = layersE
		self.generator = generator
		self.iters = 0
		
		# Initialize NNs
		self.weights, self.biases = self.initialize_NN(self.layers)
		self.Eweights, self.Ebiases = self.initialize_NN_E(self.layersE)

		with tf.name_scope("Inputs"):
			# N_inner tells us where to evaluate the function on the inside of the domain. It 
			# is a N by 2 tensor where the 0th index is t and the 1st index is x
			self.N_t = X_inner[:,0]
			self.N_x = X_inner[:,1]
			self.N_v = X_inner[:,2]
			self.X_inner = X_inner
			
			# E_inner tells us where to evaluate the electric field equation (Gauss's law) on
			# the interior of the domain
			self.E_inner = E_inner
			
			self.E_b = E_b
			self.E_t = E_t
			
			# initial conditions
			self.f_i_t = X_i[:,0]
			self.f_i_x = X_i[:,1]
			self.f_i_v = X_i[:,2]
			self.f_i = f_i
			self.X_i = X_i
			self.X_E0 = X_E0
			self.E_0 = E_0
			
			# boundary conditions
			self.f_t_t = X_t[:,0]
			self.f_t_x = X_t[:,1]
			self.f_t_v = X_t[:,2]
			self.X_t = X_t
			
			self.f_b_t = X_b[:,0]
			self.f_b_x = X_b[:,1]
			self.f_b_v = X_b[:,2]
			self.X_b = X_b
			
			# Eta evaluated
			self.X_eta = X_eta
	
		# Start a session
		self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, 
										log_device_placement=True))
   
		with tf.name_scope("Placeholders"):
			self.N__t   = tf.placeholder(tf.float32, shape=[None])
			self.N__x   = tf.placeholder(tf.float32, shape=[None])
			self.N__v   = tf.placeholder(tf.float32, shape=[None])
			self.f_i__t = tf.placeholder(tf.float32, shape=[None])
			self.f_i__x = tf.placeholder(tf.float32, shape=[None])
			self.f_i__v = tf.placeholder(tf.float32, shape=[None])
			self.f__i   = tf.placeholder(tf.float32, shape=[None])
			self.f_b__t = tf.placeholder(tf.float32, shape=[None])
			self.f_b__x = tf.placeholder(tf.float32, shape=[None])
			self.f_b__v = tf.placeholder(tf.float32, shape=[None])
			self.f_t__t = tf.placeholder(tf.float32, shape=[None])
			self.f_t__x = tf.placeholder(tf.float32, shape=[None])
			self.f_t__v = tf.placeholder(tf.float32, shape=[None])
			self.eta__t = tf.placeholder(tf.float32, shape=[None])
			self.eta__x = tf.placeholder(tf.float32, shape=[None])
			self.eta__v = tf.placeholder(tf.float32, shape=[None])
			self.E__t = tf.placeholder(tf.float32, shape=[None])
			self.E__x = tf.placeholder(tf.float32, shape=[None])
			self.E__v = tf.placeholder(tf.float32, shape=[None])
			self.E_b__t = tf.placeholder(tf.float32, shape=[None])
			self.E_b__x = tf.placeholder(tf.float32, shape=[None])
			self.E_t__t = tf.placeholder(tf.float32, shape=[None])
			self.E_t__x = tf.placeholder(tf.float32, shape=[None])
			self.E_i__t = tf.placeholder(tf.float32, shape=[None])
			self.E_i__x = tf.placeholder(tf.float32, shape=[None])
			self.E__0 = tf.placeholder(tf.float32, shape=[None])

			
		self.N_pred = self.net_N(self.N__t, self.N__x, self.N__v)
		self.f_i_pred = self.net_f(self.f_i__t, self.f_i__x, self.f_i__v)
		self.f_b_pred = self.net_f(self.f_b__t,self.f_b__x,self.f_b__v)
		self.f_t_pred = self.net_f(self.f_t__t,self.f_t__x,self.f_t__v)
		self.f_db_pred = self.net_f_d(self.f_b__t,self.f_b__x,self.f_b__v)
		self.f_dt_pred = self.net_f_d(self.f_t__t,self.f_t__x,self.f_t__v)
		self.f_d_pred = self.net_f_d(self.f_i__t,self.f_i__x,self.f_i__v)
		self.eta_pred = self.net_eta(self.eta__t,self.eta__x,self.eta__v)
		self.eta_t_pred = self.net_eta(self.f_t__t,self.f_t__x,self.f_t__v)
		self.eta_b_pred = self.net_eta(self.f_b__t,self.f_b__x,self.f_b__v)
		self.gauss_pred = self.net_gauss(self.E__t,self.E__x, self.E__v)
		self.phi_b_pred = self.net_phi(self.E_b__t, self.E_b__x)
		self.phi_t_pred = self.net_phi(self.E_t__t, self.E_t__x)
		self.E_b_pred = self.net_E(self.E_b__t, self.E_b__x)
		self.E_t_pred = self.net_E(self.E_t__t, self.E_t__x)
		self.E_db_pred = self.net_E_d(self.E_b__t, self.E_b__x)
		self.E_dt_pred = self.net_E_d(self.E_t__t, self.E_t__x)
		self.E_i_pred = self.net_E(self.E_i__t,self.E_i__x)
		
		
		
 
		
		self.NLoss = tf.reduce_mean(tf.square(self.N_pred))
		self.gaussLoss = tf.reduce_mean(tf.square(self.gauss_pred))
		self.iLoss = tf.reduce_mean(tf.square(self.f_i_pred - self.f__i))
		self.bLoss = tf.reduce_mean(tf.square(self.f_t_pred - self.f_b_pred)) +\
					 tf.reduce_mean(tf.square(self.f_dt_pred - self.f_db_pred))
					 #.5*tf.reduce_mean(tf.square(self.eta_t_pred - self.eta_b_pred))
		#self.etaLoss = tf.reduce_mean(tf.square(self.eta_pred))
		self.EbLoss = tf.reduce_mean(tf.square(self.E_b_pred - self.E_t_pred)) + \
					  tf.reduce_mean(tf.square(self.E_db_pred - self.E_dt_pred)) +\
					  tf.reduce_mean(tf.square(self.phi_b_pred - self.phi_t_pred))
		self.E0loss = tf.reduce_mean(tf.square(self.E__0 - self.E_i_pred))
		self.loss = self.iLoss + self.bLoss + self.NLoss +\
			self.gaussLoss + self.EbLoss + self.E0loss # + self.etaLoss
		
		self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss, 
																method = 'L-BFGS-B', 
																options = {'maxiter': 1000,
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
	
	def initialize_NN_E(self, layersE):        
		weights = []
		biases = []
		num_layers = len(layersE) 
		for l in range(0,num_layers-1):
			W = self.xavier_init(size=[layersE[l], layersE[l+1]])
			b = tf.Variable(tf.zeros([1,layersE[l+1]], dtype=tf.float32), dtype=tf.float32)
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
		xavier_stddev = np.sqrt(6/(in_dim + out_dim))
		return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)
	
	def forward(self, X, weights, biases, lb, ub):
		num_layers = len(weights) + 1
		H = 2.0*(X - lb)/(ub - lb) - 1.0
		for l in range(0,num_layers-2):
			W = weights[l]
			b = biases[l]
			H = tf.tanh(tf.add(tf.matmul(H, W), b))
		W = weights[-1]
		b = biases[-1]
		Y = tf.add(tf.matmul(H, W), b)
		return Y
	
	def net_eta(self, t, x, v):
		eta = self.forward(tf.stack([t,x,v],axis=1),self.weights,self.biases,self.lb,self.ub)
		return tf.squeeze(eta)
	
	def net_phi(self,t,x):
		phi = self.forward(tf.stack([t,x],axis=1),self.Eweights, self.Ebiases, self.lbE, self.ubE)
		return tf.squeeze(phi)
		
	def net_E(self,t,x):
		phi = self.net_phi(t,x)
		E = -tf.gradients(phi,x)[0]
		return E
	
	def net_E_d(self,t,x):
		phi = self.net_phi(t,x)
		E = -tf.gradients(phi,x)[0]
		E_x = tf.gradients(E, x)[0]
		return E_x
	
	def net_f(self, t, x, v):
		eta = self.net_eta(t,x,v)
		f = tf.gradients(eta, v)[0]
		return f
	
	def net_f_d(self, t, x, v):
		f = self.net_f(t,x,v)
		f_x = tf.gradients(f, x)[0]
		return f_x
	
	def net_N(self, t, x, v):
		f = self.net_f(t,x,v)
		f_t = tf.gradients(f, t)[0]
		f_x = tf.gradients(f, x)[0]
		f_v = tf.gradients(f, v)[0]
		phi = self.net_phi(t,x)
		E = -tf.gradients(phi,x)[0]
		N = f_t - E * f_v + v * f_x
		return N
	
	def net_gauss(self,t,x,v):
		phi = self.net_phi(t,x)
		E = -tf.gradients(phi,x)[0]
		E_x = tf.gradients(E, x)[0]
		n_e = self.net_eta(t,x,v) - self.net_eta(t,x,-3.5*tf.ones(tf.shape(v)[0]))
		n_i = 1.
		return 4*math.pi*(n_i - n_e) - E_x 
		
	
	def callback(self, loss, NLoss, iLoss, bLoss, gaussLoss, EbLoss, E0Loss):
		self.iters += 1
		if self.iters % 50 == 0:
			print('Loss: {}'.format(loss))
			print('Interior loss: {} Initial loss: {} Boundary loss: {}'.format(NLoss, iLoss, bLoss))
			print('E0 loss: {} Gauss\'s loss: {} E Boundary loss: {}'.format(E0Loss, gaussLoss, EbLoss))  
	
	def train(self, numSteps, N_inner, N_i, N_b, N_eta, N_E, N_Eb,N_E0):
		for i in range(numSteps):
			for X_inner, X_i, f_i, X_b, X_t, X_eta, E_inner, E_b, E_t, X_E0, E_0 in self.generator(self.X_inner, \
															  self.X_i, self.f_i, self.X_b, self.X_t, self.X_eta,\
															  self.E_inner, self.E_b, self.E_t, self.X_E0, self.E_0,\
															  N_inner, N_i, N_b, N_eta, N_E, N_Eb, N_E0):
				tf_dict = {self.N__t: X_inner[:,0], self.N__x: X_inner[:,1], self.N__v: X_inner[:,2], 
						self.f_i__t: X_i[:,0], self.f_i__x: X_i[:,1], self.f_i__v: X_i[:,2],
						self.f__i: f_i, self.f_b__t: X_b[:,0], self.f_b__x: X_b[:,1],
						self.f_b__v: X_b[:,2], self.f_t__t: X_t[:,0], self.f_t__x: X_t[:,1],
						self.f_t__v: X_t[:,2], self.eta__t: X_eta[:,0],self.eta__x: X_eta[:,1],\
						self.eta__v: X_eta[:,2], self.E__t: E_inner[:,0], self.E__x: E_inner[:,1],\
						self.E__v: E_inner[:,2],self.E_b__t: E_b[:,0], self.E_b__x: E_b[:,1], self.E_t__t: E_t[:,0],\
						self.E_t__x: E_t[:,1], self.E_i__t: X_E0[:,0], self.E_i__x: X_E0[:,1], self.E__0: E_0}
				self.optimizer.minimize(self.sess, 
								feed_dict = tf_dict,         
								fetches = [self.loss, self.NLoss, self.iLoss, self.bLoss, self.gaussLoss, self.EbLoss,self.E0loss], 
								loss_callback = self.callback)
			print(i)
			
				
	def save(self):
		tf.train.Saver().save(self.sess, "model.ckpt")
	
	def predict(self, X_star):
		f_p = self.sess.run(self.f_i_pred, {self.f_i__t: X_star[:,0], self.f_i__x: X_star[:,1], self.f_i__v: X_star[:,2]})
		d_p = self.sess.run(self.f_d_pred, {self.f_i__t: X_star[:,0], self.f_i__x: X_star[:,1], self.f_i__v: X_star[:,2]})
		N_p = self.sess.run(self.N_pred, {self.N__t: X_star[:,0], self.N__x: X_star[:,1], self.N__v: X_star[:,2]})
		eta_p = self.sess.run(self.eta_pred, {self.eta__t: X_star[:,0], self.eta__x: X_star[:,1], self.eta__v: X_star[:,2]})
		E_p = self.sess.run(self.E_b_pred, {self.E_b__t: X_star[:,0],self.E_b__x: X_star[:,1]})
		return f_p, d_p, N_p, eta_p, E_p


	def initialConditions(X, V, alpha, V_T):
		return (np.exp(-(V**2/(V_T**2)))/(math.sqrt(math.pi)*V_T)) * (1 + alpha * np.sin(2*X))
	
	def Einitial(X,alpha):
		return 4*math.pi*alpha*np.cos(2*X)/2.
	
	def generator(X_inner, X_i, f_i, X_b, X_t, X_eta, E_inner, E_b, E_t, X_E0, E_0, N_inner, N_i, N_b, N_eta, N_E, N_Eb, N_E0): 
		v_max = 3.5
		def sampleRows(X,N):
			idx = np.random.choice(np.shape(X)[0], N,replace=False)
			return X[idx,:]
		X_inner = sampleRows(X_inner,N_inner)
		I = sampleRows(np.stack([X_i[:,0],X_i[:,1],X_i[:,2],f_i],axis=1), N_i)
		X_i = I[:,0:3]
		f_i = I[:,3]
		X = sampleRows(np.stack([X_b[:,0],X_b[:,1],X_b[:,2],X_t[:,0],X_t[:,1],X_t[:,2]],axis=1), N_b)
		X_b = X[:,0:3]
		X_t = X[:,3:]
		X_inner = np.vstack([X_inner, X_i, X_b, X_t])
		X_eta = sampleRows(X_eta,N_eta)
		E_inner = sampleRows(E_inner, N_E)
		E = sampleRows(np.stack([E_b[:,0],E_b[:,1],E_t[:,0],E_t[:,1]],axis=1), N_Eb)
		E_b = E[:,0:2]
		E_t = E[:,2:]
		E_bv = np.stack((E_b[:,0],E_b[:,1],v_max*np.ones(N_Eb)),axis=1)
		E_tv = np.stack((E_t[:,0],E_t[:,1],v_max*np.ones(N_Eb)),axis=1)
		E_inner = np.vstack([E_inner, E_bv, E_tv])
		I = sampleRows(np.stack([X_E0[:,0],X_E0[:,1],E_0],axis=1), N_E0)
		X_E0 = I[:,0:2]
		E_0 = I[:,2]
		yield (X_inner, X_i, f_i, X_b, X_t, X_eta, E_inner, E_b, E_t, X_E0, E_0)
	
v_max = 3.5
t_max = 1.
alpha = 0.2
V_T = 1
pi = math.pi
x_max = pi
N_N = 100000
N_init = 150
N_pred = 150 # must be larger than N_u
N_E = 10000
N_Eb = 2000
N_Einit = 2000
layers = [3,20,20,20,20,20,20,20,1]
layersE =[2,15,15,15,15,15,1]

x = np.linspace(-x_max,x_max,N_pred)
t = np.linspace(0,t_max,N_pred)
v = np.linspace(-v_max,v_max,N_pred)

T, X, V = np.meshgrid(t,x,v)
X_star = np.hstack((T.flatten()[:,None], X.flatten()[:,None], V.flatten()[:,None]))

# Domain bounds
lb = X_star.min(0)
ub = X_star.max(0)

print(ub)
print(lb)

	
# boundary conditions
xx2 = np.stack((T[0,:,:], X[0,:,:], V[0,:,:]),axis=2) #x=-pi condition
xx2 = np.reshape(xx2,(N_pred**2,3))
xx3 = np.stack((T[-1,:,:], X[-1,:,:], V[-1,:,:]),axis=2) #x=pi condition
xx3 = np.reshape(xx3,(N_pred**2,3))
	
	
xx4 = np.stack((T[:,:,0], X[:,:,0], V[:,:,0]),axis=2)
xx4 = np.reshape(xx4,(N_pred**2,3))
plt.plot(xx4[:,0],xx4[:,1])
plt.show()
plt.plot(xx4[:,0],xx4[:,2])
plt.show()

xx1 = np.stack((np.zeros(N_init**2), np.linspace(-x_max,x_max,N_init**2), np.clip(np.squeeze(np.random.multivariate_normal([0.],[[2.]],N_init**2)),-v_max,v_max)),axis=1)
ff1 = initialConditions(xx1[:,1],xx1[:,2],alpha,V_T)
X_f_train = np.vstack([xx1, xx2, xx3])

X_N_train = lb + (ub-lb)* lhs(3, N_N)

X_N_train = np.vstack((X_N_train, X_f_train))

	
xbE = np.stack((np.linspace(0.,t_max,N_Eb),-x_max*np.ones(N_Eb)),axis=1)
xtE = np.stack((np.linspace(0.,t_max,N_Eb),x_max*np.ones(N_Eb)),axis=1)
X_E0 = np.stack((np.zeros(N_Einit),np.linspace(-x_max,x_max,N_Einit)),axis=1)
E_0 = Einitial(X_E0[:,1],alpha)
X_E = lb[0:2] + (ub[0:2]-lb[0:2])* lhs(2, N_E)
X_E = np.vstack([X_E,xbE,xtE])
X_E = np.stack([X_E[:,0],X_E[:,1],v_max*np.ones(N_E + 2*N_Eb)],axis=1)

model = vlasov1DNN(lb, ub, layers, layersE, X_N_train, X_E, xx1, ff1, xx2, xx3, xx4, xbE, xtE, X_E0, E_0, generator)

with tf.device('/device:GPU:0'):
    start_time = time.time()                
    model.train(2, 64000, 16000, 16000, 0, 4000, 1000, 1000) # N_inner, N_i, N_b, N_eta, N_E, N_Eb,N_E0
    elapsed = time.time() - start_time                
    print('Training time: %.4f' % (elapsed))

with tf.device('/device:GPU:0'):
    start_time = time.time()                
    model.train(1, 32000, 16000, 8000, 0, 8000, 2000, 2000) # N_inner, N_i, N_b, N_eta, N_E, N_Eb,N_E0
    elapsed = time.time() - start_time                
    print('Training time: %.4f' % (elapsed))

with tf.device('/device:GPU:0'):
    start_time = time.time()                
    model.train(4, 100000, 10000, 10000, 0, 10000, 2000, 2000) # N_inner, N_i, N_b, N_eta, N_E, N_Eb,N_E0
    elapsed = time.time() - start_time                
    print('Training time: %.4f' % (elapsed))

t = np.linspace(0,t_max,100)
x = np.linspace(-x_max,x_max,100)
v = np.linspace(-v_max,v_max,100)

X, V = np.meshgrid(x,v)

Xr = np.reshape(X,(10000))
Vr = np.reshape(V,(10000))

f_initial_pred1, _, _,_,_ = model.predict(np.stack((np.zeros(10000),Xr,Vr),axis=1))
f_initial_pred1 = np.reshape(f_initial_pred1, (100,100))

plt.contour(x,v,f_initial_pred1)
plt.show()
plt.imshow(np.reshape(f_initial_pred1,(100,100)))
plt.show()
plt.plot(x,f_initial_pred1[50,:])
plt.show()
plt.plot(v,f_initial_pred1[:,50])
plt.show()





f_initial_pred2, d_pred,N_pred,_, _ = model.predict(np.stack((np.ones(10000)*t_max,Xr,Vr),axis=1))
f_initial_pred2 = np.reshape(f_initial_pred2, (100,100))

plt.pcolor(x,v,f_initial_pred2 - f_initial_pred1)
plt.show()

plt.imshow(f_initial_pred2)
plt.show()
#print(model.predict(np.stack((np.zeros(10),-pi*np.ones(10),np.linspace(-v_max,v_max,10)),axis=1)))
#print(xx1[0:10,:])

T, X = np.meshgrid(t,x)

Tr = np.reshape(T,(10000))
Xr = np.reshape(X,(10000))


_,_,_,eta_pred,_ = model.predict(np.stack((Tr,Xr,-v_max*np.ones(10000)),axis=1))
eta_pred = np.reshape(eta_pred,(100,100))
plt.imshow(eta_pred)
plt.show()

_,_,_,n_e,_ = model.predict(np.stack((Tr,Xr,v_max*np.ones(10000)),axis=1))
n_e = np.reshape(n_e,(100,100))
plt.imshow(n_e)
plt.show()

plt.imshow(n_e - eta_pred)
plt.show()

plt.plot((n_e - eta_pred)[0,:])
plt.show()

plt.plot((n_e - eta_pred)[10,:])
plt.show()


plt.plot((n_e - eta_pred)[20,:])
plt.show()

plt.plot((n_e - eta_pred)[30,:])
plt.show()

_,_,_,_,E = model.predict(np.stack((np.zeros(100),np.linspace(-x_max,x_max,100),v_max*np.ones(100)),axis=1))
plt.plot(E)
plt.show()

te = 0.1
_,_,_,eta1,_ = model.predict(np.stack((te*np.ones(100),np.linspace(-x_max,x_max,100),-v_max*np.ones(100)),axis=1))
plt.plot(eta1)
plt.show()

_,_,_,eta2,_ = model.predict(np.stack((te*np.ones(100),np.linspace(-x_max,x_max,100),v_max*np.ones(100)),axis=1))
plt.plot(eta2)
plt.show()

plt.plot(eta2-eta1)
plt.show()


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

    T = np.ones(10000) * t_max
    t = 0.
    
    Xr = np.reshape(X,(10000))
    Vr = np.reshape(V,(10000))
    f_initial_pred, _, _, _,_ = model.predict(np.stack((t*T,Xr,Vr),axis=1))
    f_initial_pred = np.reshape(f_initial_pred, (100,100))
    
    im = ax.imshow(f_initial_pred,interpolation='nearest',origin='lower')
    
    fig.set_size_inches([8,4])


    tight_layout()


    def update_img(n):
        f_pred, _, _, _,_ = model.predict(np.stack((n*T/100. + 0.01,Xr,Vr),axis=1))
        f_pred = np.reshape(f_pred, (100,100))
        im.set_data(f_pred)
        return im

    #legend(loc=0)
    ani = animation.FuncAnimation(fig,update_img,100,interval=30)
    writer = animation.writers['ffmpeg'](fps=30)

    ani.save('vlasov1D.mp4',writer=writer,dpi=dpi)
    return ani

ani_frame(model)
print("Done")

import matplotlib.animation as animation
import numpy as np
from pylab import *


dpi = 100

    


fig, ax = plt.subplots()
t=0.
num=1000
T = np.ones(num) * t_max
X= np.linspace(-x_max,x_max,num)
V = v_max*np.ones(num)

_, _, _, _, E = model.predict(np.stack((t*T,X,V),axis=1))
line, = ax.plot(E)

def init():  # only required for blitting to give a clean slate.
    _, _, _, _, E = model.predict(np.stack((t*T,X,V),axis=1))
    line.set_ydata(E)
    return line,

def animate(i):
    _, _, _, _, E = model.predict(np.stack((i*T/100.,X,V),axis=1))
    line.set_ydata(E)  # update the data.
    return line,

ani = animation.FuncAnimation(fig, animate, init_func=init, interval=2, blit=True, save_count=100)
writer = animation.writers['ffmpeg'](fps=30)
ani.save('vlasov1D_Efield.mp4',writer=writer,dpi=dpi)
print("Done")

import matplotlib.animation as animation
import numpy as np
from pylab import *


dpi = 100

    


fig, ax = plt.subplots()
t=0.
num=1000
T = np.ones(num) * t_max
X= np.linspace(-x_max,x_max,num)
Vmax = v_max*np.ones(num)
Vmin = -v_max*np.ones(num)

_, _, eta_pred, _, _ = model.predict(np.stack((t*T,X,Vmin),axis=1))
_, _, n_e, _, _ = model.predict(np.stack((t*T,X,Vmax),axis=1))
line, = ax.plot(n_e - eta_pred)

def init():  # only required for blitting to give a clean slate.
    _, _, eta_pred, _, _ = model.predict(np.stack((t*T,X,Vmin),axis=1))
    _, _, n_e, _, _ = model.predict(np.stack((t*T,X,Vmax),axis=1))
    line.set_ydata(n_e - eta_pred)
    return line,

def animate(i):
    _, _, eta_pred, _, _ = model.predict(np.stack((i*T/100.,X,Vmin),axis=1))
    _, _, n_e, _, _ = model.predict(np.stack((i*T/100.,X,Vmax),axis=1))
    line.set_ydata(n_e - eta_pred)  # update the data.
    return line,

ani = animation.FuncAnimation(fig, animate, init_func=init, interval=2, blit=True, save_count=100)
writer = animation.writers['ffmpeg'](fps=30)
ani.save('vlasov1D_n_e_field.mp4',writer=writer,dpi=dpi)
print("Done")