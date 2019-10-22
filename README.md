# Physics-informed neural networks (PINNs)

Nick McGreivy and Phil Travis are working on solving some problems in Plasma Physics with Physics-Informed Neural Networks (PINNs). This work is no longer in progress, but is being presented at APS-DPP 2019.

### Resources
This project is an application of Maziar Rassi's "physics informed neural networks" to plasma physics problems. The original paper by Rassi can be found on [arXiv:1711.10561](https://arxiv.org/abs/1711.10561). Rassi also has a blog post explaining the concept [here](http://www.dam.brown.edu/people/mraissi/research/1_physics_informed_neural_networks/).

The poster given at APS DPP 2019 is in the root directory as `APS-Poster-McGreivy-2019.pdf`.

### Requirements
Code requires:

* Python 3.x
* TensorFlow 1.14 (*not* 2.0 or above, unsure about the lower bound--Google Colab should work fine)
* ... and other common libraries

### Code examples
`vlasovFreeStream.py`: solving the free-streaming Vlasov equation, $\frac{\partial f}{\partial t} + v \frac{\partial f}{\partial x}$, using physics informed neural networks. __This is the most commented / documented code file.__ This model converges to a solution nicely.

`boundaryLayer.py`: solving a boundary layer problem, $$\epsilon \frac{\partial^2 u}{\partial x^2 } + (1+x)\frac{\partial u}{\partial x} + u=0 $$, for small epsilon. In this case, it is solved for $$\epsilon = 0.01$$ and $$0.005$$. This NN also converges.

`vlasovEfield.py`: Attempting to solve the Vlasov equation, $$\frac{\partial f}{\partial t} + v \frac{\partial f}{\partial x} + \frac{q}{m} E \frac{\partial f}{\partial v}$$, now with an electric field term whose solution is given by Gauss's law. The solution did not converge within a reasonable amount of time and model capacity. 

Here the solution $$f$$ is being represented by a neural network whose inputs are $$x$$, $$v$$, and $$t$$. The electric field is represented by a different neural network whose inputs are $$x$$ and $$t$$. 

`fluidOscillations.py:`: Here we solve a set of cold-plasma fluid equations, which are continuity and momentum equations with an electric field term. These equations are $$\frac{\partial n}{\partial t} + \frac{\partial}{\partial x}(nu)=0$$, $$\frac{\partial u}{\partial t} + u \frac{\partial u}{\partial x} = -e E$$, and $$e(1 - n) = \frac{\partial E}{\partial x}$$. 
