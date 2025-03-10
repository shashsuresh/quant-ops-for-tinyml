# Quantized operations for TinyML

A collection of quantized versions of commonly found NN layers, designed for simulation of MCU inference and training. This repository is based on MIT HAN Lab's work in [Tiny Training](https://github.com/mit-han-lab/tiny-training).

## Pre-requisites

Here are some python packages that must be installed to use this repository.

- PyTorch
- NumPy

## Structure

- `quantized_operations`
- `conversions.py`
- `quantized_network_builder.py`
- `utils.py`

## Including in other projects

This package can be imported as a Git submodule in other repositories. The operations can then be imported as follows:
`from <NAME OF SUBMODULE>.quantized_operations import <OPERATION YOU WISH TO INCLUDE>`.