# **Mo**deling **Cha**os with **N**eural **N**etworks

A set of scripts for modeling chaotic systems using neural networks, built with Pytorch/Lightning.

Much of the code in this repo is derived from [Nixtla's neuralforecast repo](https://github.com/Nixtla/neuralforecast), although it has been heavily modified.  The main changes include: removing the dependency on Pandas and rewriting data pipelines in order to reduce cpu and gpu memory requirements, removing certain other dependencies (Jupyter, Ray), and simplifying the implementation by restricting the supported use cases to those that are relevant to this project's goals.
 
