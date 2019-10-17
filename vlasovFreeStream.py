import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
# import math
from pyDOE import lhs
# from scipy.interpolate import griddata
# import scipy
# import scipy.io

import time

# Weights and biases
import wandb
wandb.init(project='pinn', entity="nick-and-phil")

np.random.seed(42)
tf.set_random_seed(42)


class vlasovFreeStreamNN:

    def __init__(self, lb, ub, layers, X_inner, X_i, u_i, X_b, X_t, generator):

        # Lower bounds of t, x, v
        self.lb = lb
        # Upper bounds of t, x, v
        self.ub = ub
        # Number of layers in the neural network
        self.layers = layers
        # Function to use for generating training point samples
        self.generator = generator
        # Number of training iterations
        self.iters = 0

        # Initialize NNs
        self.weights, self.biases = self.initialize_NN(layers)

        with tf.name_scope("Inputs"):
            # f_inner tells us where to evaluate the function on the inside of the domain. It
            # is a N by 2 tensor where the 0th index is t and the 1st index is x
            # We want f = 0
            self.f_t = X_inner[:, 0]
            self.f_x = X_inner[:, 1]
            self.f_v = X_inner[:, 2]
            # X_inner is the interior sampling points
            self.X_inner = X_inner

            # initial conditions
            # self.u_i_t = X_i[:, 0]
            # self.u_i_x = X_i[:, 1]
            # self.u_i_v = X_i[:, 2]
            self.u_i = u_i
            self.X_i = X_i

            # self.u_t_t = X_t[:, 0]
            # self.u_t_x = X_t[:, 1]
            # self.u_t_v = X_t[:, 2]
            self.X_t = X_t

            # self.u_b_t = X_b[:, 0]
            # self.u_b_x = X_b[:, 1]
            # self.u_b_v = X_b[:, 2]
            self.X_b = X_b

        # Start a session
        tfconfig = tf.ConfigProto()
        # Set the session to use only the memory it needs so you can run concurrent experiments
        #   on a single GPU
        tfconfig.gpu_options.allow_growth = True
        tfconfig.allow_soft_placement = True
        # tf.config.log_device_placement = True
        self.sess = tf.Session(config=tfconfig)

        with tf.name_scope("Placeholders"):
            # Tensors for f evaluation from interior points
            self.f__t = tf.placeholder(tf.float32, shape=[None])
            self.f__x = tf.placeholder(tf.float32, shape=[None])
            self.f__v = tf.placeholder(tf.float32, shape=[None])
            # Tenors for initial condition evaluation?
            self.u_i__t = tf.placeholder(tf.float32, shape=[None])
            self.u_i__x = tf.placeholder(tf.float32, shape=[None])
            self.u_i__v = tf.placeholder(tf.float32, shape=[None])
            self.u__i = tf.placeholder(tf.float32, shape=[None])
            # Tensors for boundary condition evaluation
            self.u_b__t = tf.placeholder(tf.float32, shape=[None])
            self.u_b__x = tf.placeholder(tf.float32, shape=[None])
            self.u_b__v = tf.placeholder(tf.float32, shape=[None])
            self.u_t__t = tf.placeholder(tf.float32, shape=[None])
            self.u_t__x = tf.placeholder(tf.float32, shape=[None])
            self.u_t__v = tf.placeholder(tf.float32, shape=[None])

        # The prediction of f made by the NN by taking derivatives of u
        self.f_pred = self.net_f(self.f__t, self.f__x, self.f__v)

        # Prediction of u made by the NN. I think these are the same thing
        self.u_pred = self.net_u(self.u_i__t, self.u_i__x, self.u_i__v)
        self.u_init_pred = self.net_u(self.u_i__t, self.u_i__x, self.u_i__v)

        # x minimum boundary conditions
        self.u_b_pred = self.net_u(self.u_b__t, self.u_b__x, self.u_b__v)
        # x maximum boundary conditions
        self.u_t_pred = self.net_u(self.u_t__t, self.u_t__x, self.u_t__v)
        # Derivatives of u for boundary conditions (b and t) and initial conditions (I think)
        self.u_db_pred = self.net_u_d(self.u_b__t, self.u_b__x, self.u_b__v)
        self.u_dt_pred = self.net_u_d(self.u_t__t, self.u_t__x, self.u_t__v)
        self.u_d_pred = self.net_u_d(self.u_i__t, self.u_i__x, self.u_i__v)

        # The goal is to get f=0 everywhere
        self.fLoss = tf.reduce_mean(tf.square(self.f_pred))
        # Match initial conditions predicted by the network with real ones
        self.iLoss = tf.reduce_mean(tf.square(self.u_init_pred - self.u__i))
        # Match value and derivatives at the boundaries
        self.bLoss = (0.5 * tf.reduce_mean(tf.square(self.u_t_pred - self.u_b_pred)) +
                      0.5 * tf.reduce_mean(tf.square(self.u_dt_pred - self.u_db_pred)))
        # Total loss
        self.loss = self.iLoss + self.fLoss + self.bLoss
        # Need to use 2nd order optimization to minimize loss that has derivatives
        # Optimization stops when ftol is reached or after maxiter iterations or after maxfun
        #   function calls
        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss,
                                                                method='L-BFGS-B',
                                                                options={'maxiter': 40000,
                                                                         'maxfun': 100000,
                                                                         'maxcor': 50,
                                                                         'maxls': 50,
                                                                         'ftol': 1.0 * np.finfo(float).eps})

        init = tf.global_variables_initializer()
        self.sess.run(init)

    def initialize_NN(self, layers):
        weights = []
        biases = []
        num_layers = len(layers)
        for l in range(0, num_layers - 1):
            W = self.xavier_init(size=[layers[l], layers[l + 1]])
            b = tf.Variable(tf.zeros([1, layers[l + 1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)
        return weights, biases

    def kaiming_init(self, size):
        in_dim = size[0]
        out_dim = size[1]
        kaiming_stddev = np.sqrt(2 / (in_dim))
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=kaiming_stddev), dtype=tf.float32)

    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = np.sqrt(2 / (in_dim + out_dim))
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)

    # NN forward propagation
    def forward(self, X, weights, biases):
        num_layers = len(weights) + 1
        # Normalize between 0 and 1?
        H = 2.0 * (X - self.lb) / (self.ub - self.lb) - 1.0
        for l in range(0, num_layers - 2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y

    # Use the NN to predict the value of u from t, x, v stack
    def net_u(self, t, x, v):
        u = self.forward(tf.stack([t, x, v], axis=1), self.weights, self.biases)
        return tf.squeeze(u)

    # Calculate gradient of u with respect to x
    def net_u_d(self, t, x, v):
        u = self.net_u(t, x, v)
        u_x = tf.gradients(u, x)[0]
        return u_x

    # Calculate f (which =0). This is the collision- and force-free version of the vlasov equation
    def net_f(self, t, x, v):
        u = self.net_u(t, x, v)
        u_t = tf.gradients(u, t)[0]
        u_x = tf.gradients(u, x)[0]
        f = u_t + v * u_x
        return f

    # Need this to get loss information out of the optimization routine
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

    # Train the NN. numSteps controls the number of iterations through 
    def train(self, numSteps, N_inner, N_i, N_b):
        for i in range(numSteps):
            for X_inner, X_i, u_i, X_b, X_t in self.generator(self.X_inner, self.X_i, self.u_i,
                                                              self.X_b, self.X_t,
                                                              N_inner, N_i, N_b):
                print(X_inner[:, 0])
                print("\n\n\n")
                tf_dict = {self.f__t: X_inner[:, 0],
                           self.f__x: X_inner[:, 1],
                           self.f__v: X_inner[:, 2],
                           self.u_i__t: X_i[:, 0],
                           self.u_i__x: X_i[:, 1],
                           self.u_i__v: X_i[:, 2],
                           self.u__i: u_i,
                           self.u_b__t: X_b[:, 0],
                           self.u_b__x: X_b[:, 1],
                           self.u_b__v: X_b[:, 2],
                           self.u_t__t: X_t[:, 0],
                           self.u_t__x: X_t[:, 1],
                           self.u_t__v: X_t[:, 2]}
                self.optimizer.minimize(self.sess,
                                        feed_dict=tf_dict,
                                        fetches=[self.loss, self.fLoss, self.iLoss, self.bLoss],
                                        loss_callback=self.callback)
            print(i)

    def save(self):
        tf.train.Saver().save(self.sess, "model.ckpt")

    def predict(self, X_star):
        u_p = self.sess.run(
            self.u_pred, {self.u_i__t: X_star[:, 0], self.u_i__x: X_star[:, 1], self.u_i__v: X_star[:, 2]})
        d_p = self.sess.run(
            self.u_d_pred, {self.u_i__t: X_star[:, 0], self.u_i__x: X_star[:, 1], self.u_i__v: X_star[:, 2]})
        f_p = self.sess.run(
            self.f_pred, {self.f__t: X_star[:, 0], self.f__x: X_star[:, 1], self.f__v: X_star[:, 2]})
        return u_p, d_p, f_p


def initialConditions(X, V, alpha):
    return np.exp(-(V ** 2)) * (1 + alpha * np.sin(2 * X))


# Generate samples of inner points, initial conditions, and boundary conditions
def generator(X_inner, X_i, u_i, X_b, X_t, N_inner, N_i, N_b):
    def sampleRows(X, N):
        idx = np.random.choice(np.shape(X)[0], N, replace=False)
        return X[idx, :]
    X_inner = sampleRows(X_inner, N_inner)
    I = sampleRows(np.stack([X_i[:, 0], X_i[:, 1], X_i[:, 2], u_i], axis=1), N_i)
    X_i = I[:, 0:3]
    u_i = I[:, 3]
    X = sampleRows(np.stack([X_b[:, 0], X_b[:, 1], X_b[:, 2],
                             X_t[:, 0], X_t[:, 1], X_t[:, 2]], axis=1), N_b)
    X_b = X[:, 0:3]
    X_t = X[:, 3:]
    X_inner = np.vstack([X_inner, X_i, X_b, X_t])
    yield (X_inner, X_i, u_i, X_b, X_t)


pi = np.pi  # bruh

# t_max is the range of the time grid. Time always starts at 0
t_max = 2 * pi
# x_max is the one-sided range of the position grid
x_max = pi
# vmax is the one-sided range of the velocity grid
v_max = 3
# alpha controls the scale of the sin feature in position
alpha = 0.25

# Number of points to inspect / train on? 
N_f = 1000000
# Number of initial conditions? 
N_init = 300
# Number of different values of time, position, and velociy. Number of predicted points? 
N_pred = 300  # must be larger than N_u (what is N_u?) 
# The number of neurons in each layer of the network
layers = [3, 20, 20, 20, 20, 20, 20, 20, 1]

# Create the range of values for t, x, and v
t = np.linspace(0, t_max, N_pred)
x = np.linspace(-x_max, x_max, N_pred)
v = np.linspace(-v_max, v_max, N_pred)

# Create grid of all combinations of T, X, and V values (3D)
T, X, V = np.meshgrid(t, x, v)
# What is X_star 
# Has shape (N_pred ** 3, 3)
X_star = np.hstack((T.flatten()[:, None], X.flatten()[:, None], V.flatten()[:, None]))

# Domain bounds for t, x, and v. Equivelant to taking +/- of largest {t,x,v}_max value (I think)
lb = X_star.min(0)
ub = X_star.max(0)

# Boundary conditions at x = -x_max (that's why the first index is 0?)
xx2 = np.stack((T[0, :, :], X[0, :, :], V[0, :, :]), axis=2)
# xx2 has shape (N_pred, N_pred, 3). Reshape:
xx2 = np.reshape(xx2, (N_pred ** 2, 3))
# Boundary conditions at x = x_max (that's why the first index is -1?)
xx3 = np.stack((T[-1, :, :], X[-1, :, :], V[-1, :, :]), axis=2)
xx3 = np.reshape(xx3, (N_pred ** 2, 3))

# Use latin-hypercube sampling to get N_pred ** 2 random smaples of (x, v)
# This makes the interior sampling points (I think)
X_inner = lb[1:3] + (ub[1:3] - lb[1:3]) * lhs(2, N_pred ** 2)

# No idea what the xx1 or uu1 are
# xx1 is a stack of arrays of 0's, position, and a guassian used to calculate the initial conditions
xx1 = np.stack((np.zeros(N_init ** 2),
                np.linspace(-x_max, x_max, N_init ** 2),
                np.clip(np.squeeze(np.random.multivariate_normal([0], [[1.5]], N_init ** 2)),
                        -3, 3)), axis=1)

# Calcualte the initial from an array of points (xx1)
uu1 = initialConditions(xx1[:, 1], xx1[:, 2], alpha)  # exp(-v^2) (1 + alpha sin(2x))

# xx1: stack of points used to calcualte initial conditions
# xx2: minimum x boundary conditions
# xx3: maximum x boundary conditions
X_u_train = np.vstack([xx1, xx2, xx3])

# Use latin-hypercube sampling to get an array of random times, positions, and velocities
# I assume N_f is the number of points to evaluate f at
X_f_train = lb + (ub - lb) * lhs(3, N_f)

# Stack N_f random time-position-velocity pairs with initial and boundary conditions
X_f_train = np.vstack((X_f_train, X_u_train))

# Build the neural network
model = vlasovFreeStreamNN(lb, ub, layers, X_f_train, xx1, uu1, xx2, xx3, generator)

with tf.device('/device:GPU:0'):
    start_time = time.time()
    # Number of steps, number of interior evaluation points, number of initial conditions,
    #   and lastly, number of boundary conditions
    model.train(2, 10000, 1000, 1000)
    elapsed = time.time() - start_time
    print('Training time: %.4f' % (elapsed))

########################
# Lets plot some stuff #
########################
t = np.linspace(0, t_max, 100)
x = np.linspace(-pi, pi, 100)
v = np.linspace(-v_max, v_max, 100)

X, V = np.meshgrid(x, v)

Xr = np.reshape(X, (10000))
Vr = np.reshape(V, (10000))

print(np.shape(X))
u_initial_pred1, _, _ = model.predict(np.stack((np.zeros(10000), Xr, Vr), axis=1))
u_initial_pred1 = np.reshape(u_initial_pred1, (100, 100))

plt.contour(x, v, u_initial_pred1)
plt.show()
plt.imshow(np.reshape(u_initial_pred1, (100, 100)))
plt.show()
plt.plot(x, u_initial_pred1[50, :])
plt.show()
plt.plot(v, u_initial_pred1[:, 50])
plt.show()

u_initial_pred2, d_pred, f_pred = model.predict(np.stack((np.ones(10000), Xr, Vr), axis=1))
print(np.shape(f_pred))
print(np.shape(d_pred))
u_initial_pred2 = np.reshape(u_initial_pred2, (100, 100))

print(np.shape(d_pred))
print(np.shape(f_pred))

plt.pcolor(x, v, u_initial_pred2 - u_initial_pred1)
plt.show()

plt.imshow(u_initial_pred2)
plt.show()
print(model.predict(np.stack((np.zeros(10), -pi * np.ones(10), np.linspace(-v_max, v_max, 10)), axis=1)))
print(xx1[0:10, :])

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

    x = np.linspace(-pi, pi, 100)
    v = np.linspace(-v_max, v_max, 100)

    X, V = np.meshgrid(x, v)

    T = t_max * np.ones(10000)
    t = 0.

    Xr = np.reshape(X, (10000))
    Vr = np.reshape(V, (10000))
    u_initial_pred, _, _ = model.predict(np.stack((t * T, Xr, Vr), axis=1))
    u_initial_pred = np.reshape(u_initial_pred, (100, 100))

    im = ax.imshow(u_initial_pred, interpolation='nearest', origin='lower')

    fig.set_size_inches([8, 4])

    tight_layout()

    def update_img(n):
        u_pred, _, _ = model.predict(np.stack((n * T / 100. + 0.01, Xr, Vr), axis=1))
        u_pred = np.reshape(u_pred, (100, 100))
        im.set_data(u_pred)
        return im

    # legend(loc=0)
    ani = animation.FuncAnimation(fig, update_img, 100, interval=30)
    writer = animation.writers['ffmpeg'](fps=30)

    ani.save('vlasovWorking.mp4', writer=writer, dpi=dpi)
    return ani


ani_frame(model)
print("Done")
