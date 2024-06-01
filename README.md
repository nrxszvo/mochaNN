# Modeling Chaos with Neural Networks

A set of scripts for modeling chaotic dynamical systems using neural networks, built with Pytorch/Lightning.

See [environment.yml](https://github.com/nrxszvo/mochaNN/blob/main/environment.yml) for dependencies.

Currently only the [NHiTS](https://arxiv.org/abs/2201.12886) neural architecture is supported. The NHiTS model code is adopted from [Nixtla's neuralforecast repo](https://github.com/Nixtla/neuralforecast), with some minor modifications.

The [dysts](https://github.com/williamgilpin/dysts) repo is used as a reference how to generate datasets for choatic systems.  Currently the original Lorenz Attractor ("Lorenz 63") is the only dynamical system that is supported by the data generation scripts, but all of the code can easily be extended to support arbitrary systems.

See [Generic Deep Learning for Chaotic Dyanmics: A Case Study on the Lorenz Attractor](https://nrxszvo.github.io/nhits-lorenz) for an example of a research project that used this repo.
 
