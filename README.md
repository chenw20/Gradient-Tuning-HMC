# Hamiltonian Measure Preverving Flows
This is the demo code of Hamiltonian Measure Preserving Flows

## Requirement
To run this demo code, you need the following python package installed
- Python 3.5 (python 3 is necessary, but not exact version)
- Tensorflow 1.5.0 (GPU support is optional, but strongly recommended)
- Numpy 1.14
- Matplotlib 2.1.0

Here is the link of installation of Tensorflow GPU with python virtual
environment:
https://www.tensorflow.org/install/install_linux#InstallingVirtualenv

## What's in here
The main purpose of this project is to demonstrate Hamiltonian Measure
Preverving Flows (HMPFs), which is a variant of Measure Preverving Flows
, a general variational inference method developed for statistical
machine learning. But, it can be used in other research fields involved
with statistical modelling.

You can find the paper on arxiv:
https://arxiv.org/abs/1805.10377

The code here is just for demo purpose only.

The list of demos in the paper:

- Training and visualise HMPFs on bivariate distributions

- Estimation of variational lower bound and marginal likelihood based on
 HMPFs on a deep convolutional generative model (DCGM) with MNIST.

The list of coming-ups:

- Evaluation of HMPFs for Bayesian neural networks

Other demos:

- Many people have the concern of the volume preservation of Leapfrog
algorithm. See here: https://twitter.com/dustinvtran/status/1001329474678415361.
Please read the tutorial of Radford Neal
(https://arxiv.org/pdf/1206.1901.pdf) for the explanation.
To provide a numeric evidence of the exact preserveing volume of the
Leapfrog algorithm, you can check the script jacobian_test.py

## How to run it:

- To run the demo of 2d distributions: demo_2d.py
  - The code of available 2d distributions is in package targets

- To run the demo of DCGM on MNIST : vae_eval.py
  - The code of network decoder is in package decoder
  - Two versions in this demo on:
    - The Tensorflow MNIST with online dynamical binarisation
    - Prebinarised MNIST from IWAE Gibhub repository
    (the number reported in our paper)
    https://github.com/yburda/iwae/tree/master/datasets/BinaryMNIST



