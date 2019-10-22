# Physics-informed neural networks (PINNs)

Nick McGreivy and Phil Travis are working on solving some problems in Plasma Physics with Physics-Informed Neural Networks (PINNs). This is in progress and will be updated in preparation for APS-DPP 2019.

### Resources
This project is an application of Maziar Rassi's "physics informed neural networks" to plasma physics problems. The original paper by Rassi can be found on [arXiv:1711.10561](https://arxiv.org/abs/1711.10561). Rassi also has a blog post explaining the concept [here](http://www.dam.brown.edu/people/mraissi/research/1_physics_informed_neural_networks/).

The poster given at APS DPP 2019 is in the root directory as `APS-Poster-McGreivy-2019.pdf`

### Requirements
Code requires:

* Python 3.x
* TensorFlow 1.14 (*not* 2.0 or above, unsure about the lower bound--Google Colab should work fine)
* ... and other common libraries

### Code examples
`vlasovFreeStream.py`: solving the free-streaming Vlasov equation using physics informed neural networks. __This is the most commented / documented code file.__ This model converges to a solution nicely.

`boundaryLayer.py`: solving a boundary layer problem. This NN also converges.

`vlasovEfield.py`: Attempting to solve the Vlasov equation with an electric field term. Solution did not converge within a reasonable amount of time and model capacity.

`fluidOscillations.py`: ¯\\\_(ツ)_/¯ 