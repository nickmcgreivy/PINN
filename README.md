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
`vlasovFreeStream.py`: solving the free-streaming Vlasov equation

![equation](https://latex.codecogs.com/png.latex?\dpi{200}&space;\normal&space;\frac{\partial&space;f}{\partial&space;t}&space;&plus;&space;v&space;\frac{\partial&space;f}{\partial&space;x}=0)

using physics informed neural networks. __This is the most commented / documented code file.__ This model converges to a solution nicely.

`boundaryLayer.py`: solving a boundary layer problem

![equation](https://latex.codecogs.com/png.latex?\dpi{200}&space;\epsilon&space;\frac{\partial^2&space;u}{\partial&space;x^2&space;}&space;&plus;&space;(1&plus;x)\frac{\partial&space;u}{\partial&space;x}&space;&plus;&space;u=0)

for small epsilon. In this case, it is solved for $$\epsilon = 0.01$$ and $$0.005$$. This NN also converges.

`vlasovEfield.py`: Attempting to solve the Vlasov equation

![equation](https://latex.codecogs.com/png.latex?\dpi{200}&space;\frac{\partial&space;f}{\partial&space;t}&space;&plus;&space;v&space;\frac{\partial&space;f}{\partial&space;x}&space;&plus;&space;\frac{q}{m}&space;E&space;\frac{\partial&space;f}{\partial&space;v}=0)

now with an electric field term whose solution is given by Gauss's law. The solution did not converge within a reasonable amount of time and model capacity. 

Here the solution f is being represented by a neural network whose inputs are x, v, and t. The electric field is represented by a different neural network whose inputs are x and t. 

`fluidOscillations.py:`: Here we solve a set of cold-plasma fluid equations, which are continuity and momentum equations with an electric field term. These equations are 

![equation](https://latex.codecogs.com/png.latex?\dpi{200}&space;\frac{\partial&space;n}{\partial&space;t}&space;&plus;&space;\frac{\partial}{\partial&space;x}(nu)=0)

![equation](https://latex.codecogs.com/png.latex?\dpi{200}&space;\frac{\partial&space;u}{\partial&space;t}&space;&plus;&space;u&space;\frac{\partial&space;u}{\partial&space;x}&space;=&space;-e&space;E)

![equation](https://latex.codecogs.com/png.latex?\dpi{200}&space;e(1&space;-&space;n)&space;=\epsilon_0&space;\frac{\partial&space;E}{\partial&space;x})
