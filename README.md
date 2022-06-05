# Multi-Augmentation Self-Supervised Visual Representation Learning 

<p align="center">
  <img src="images/MASSRL.gif" width="400">
</p>
<p align="center">
<font size=0.7> MASSRL Self-Supervised Pretraining Framework.</font> 
</p>

This repo is official Pytorch-Lightning implementation MASSRL.

[MASSRL Paper link](https://arxiv.org/abs/1911.05722)

[Blog Post]("Coming Soon")

```
MASSRL leverages the multi-Augmentation strategy to allow us to increase the quantity and diversity of data.
The result shows that requiring fewer epochs of iteration training can perform significantly better than the baseline
like (BYOL or SimCLR) for the self-supervised pre-training stage. 
```
<p align="center">
  <img src="images/pre-training_stage.png" width="380">
</p>

<font size="0.5"> *Figure: Comparsion effective of the learned representation during self-supervised pre-training stage between BYOL (Grill, Strub et al. 2020) & MA-SSRL (ours) use the same ResNet-50 backbone (IN) in Top-1 accuracy(%).*</font> 


This repo contains the source code for the `MASSRL`, our tool that makes the implementation of multi-Augmentation Strategies in Pytorch models effortless and less error-prone.

## Table of Contents

  - [Installation](#installation)
  - [Visualization `MASSRL` Multi-Augmentation Strategies ](#Different-Multi-Augmentation-properties)
  - [Configure Self-Supervised Pretraining](#Setup-self-supervised-pretraining)
    - [Dataset](#Natural-Image-Dataset)
    - [Hyperamters Setting](#Important-Hyperparameter-Setting)
    - [Choosing # augmentation Strategies](#Number-Augmentation-Strategies)
    - [Single or Multi GPUs](#Single-Multi-GPUS)
  - [Examples](#examples)
  - [Downstream Tasks](#running-tests)
     - [Image Classification Tasks](#Natural-Image-Classification)
     - [Other Vision Tasks](#Object-Detection-Segmentation)
  - [Current Limitations](#current-limitations)
  - [Contributing](#contributing)


## Installation

```
pip or conda installs these dependents in your local machine
```
### Requirements
* torch
* torchvision
* tqdm
* einops
* wandb
* pytorch-lightning
* lightning-bolts
* torchmetrics
* scipy
* timm

**Optional**:
* nvidia-dali
* matplotlib
* seaborn
* pandas
* umap-learn

## Visualization `MASSRL` Multi-Augmentation Strategies

<a target="[_parent](https://colab.research.google.com/drive/1fquGOr_psJfDXxOmdFVkfrbedGfi1t-X?usp=sharing)"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
Visualization Multi-Augmentation Strategies on Google-Colab Notebook: https://colab.research.google.com/drive/1fquGOr_psJfDXxOmdFVkfrbedGfi1t-X?usp=sharing 

Note the Visualization Augmentation *do not need to be trained* --- we are only Visualize Image after apply different Augmentation transformations.
However, you need to make sure that the `dataset` is appropriately passed down to the constructor of all submodules.
If you want to see this happen, please upvote [this Repo issue]

## Configure Self-Supervised `Pretraining` Stage

For pretraining the backbone, follow one of the many bash files in `bash_files/pretrain/`.

After that, for offline linear evaluation, follow the examples in `bash_files/linear`.

There are extra experiments on K-NN evaluation in `bash_files/knn/` and feature visualization with UMAP in `bash_files/umap/`.

**NOTE:** Files try to be up-to-date and follow as closely as possible the recommended parameters of each paper, but check them before running.



For example, the `mup.coord_check.example_plot_coord_check` function is implemented this way for toy MLP and CNN models.


#### Tips for Coord Check

- Use a large learning rate (larger than you'd use for actual training). This would emphasize any potential exploding coordinates issue, which could be hidden by the initialization if the learning rate is too small.
- If you reuse a module multiple times in the forward pass, then `mup.get_coord_data` will only record the statistics from the last usage. In this case, for testing purposes, one can wrap different usages with `nn.Identity` modules of different names to distinguish them.

### Wider is Always Better

![](figures/widerbetter.png)

Another sign that μP has not been implemented correctly is if going wider does worse (on training loss) after some width, at some point during training.
The figure above illustrates this in a collection of training curves: (left) the correct implementation should always see performance improve with width, at any point in training; (middle) if you used standard parametrization (SP), sometimes you may see performance improve with width up to some point and then suddenly it becomes worse with wider models; (right) or you may immediately see worsening performance even for narrow models.

## Examples
See the `MLP`, `Transformer`, and `ResNet` folders inside `examples/` as well as the tests in `mup/test` for examples.
People familiar with [Huggingface Transformers](https://github.com/huggingface/transformers) may also find the `examples/mutransformers` submodule instructive (obtained via `git submodule update --init`), which is also available standalone at [https://github.com/microsoft/mutransformers](https://github.com/microsoft/mutransformers).

## Current Limitations

- `set_base_shapes(model, ...)` assumes that `model` has just been randomly initialized in the standard way and rescales its parameters using the base shape information so the model is in μP.
- If you want data parallelism, please use `torch.nn.parallel.DistributedDataParallel` instead of `torch.nn.DataParallel`. This is because the latter removes the 


## Native Integration With Huggingface

Frustrated that your [Huggingface Transformer](https://github.com/huggingface/transformers) breaks when you scale up? Want to tune hyperparameters for your large mult-GPU [Huggingface Transformer](https://github.com/huggingface/transformers) on a single GPU, right out the box? If so, please upvote [this github issue](https://github.com/huggingface/transformers/issues/16157)!


## Running Tests
To run tests, do
```bash
python -m mup.test
```


## The Basic Math

μP is designed so as to satisfy the following desiderata:

> At any time during training
> 1. Every (pre)activation vector in a network should have Θ(1)-sized coordinates
> 2. Neural network output should be O(1).
> 3. All parameters should be updated as much as possible (in terms of scaling in width) without leading to divergence

It turns out these desiderata uniquely single out μP.
To derive μP from them, one needs to carefully consider how the *coordinate size* of a vector Av, resulting from a square matrix A multiplying vector v, depends on those of A and v, when A and v are "correlated".
Those of type 1 cover things like weight gradients; those of type 2 cover things like weight initialization.
Then, if A and v both have entry size Θ(1) and they are correlated in ways that arise naturally during training, then we have the following table.

|                  | outer product A (type 1) | iid A  (type 2)    |
|------------------|--------------------------|--------------------|
| Entry size of Av | Θ(n)                     | Θ(sqrt(n))         |

Given this table, one can then trace the forward and backward computation of a network to derive μP straightforwardly.

See [our blog post](https://www.microsoft.com/en-us/research/blog/%C2%B5transfer-a-technique-for-hyperparameter-tuning-of-enormous-neural-networks/) for a gentle primer and [our paper](https://arxiv.org/abs/2203.03466) for details.


## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details,

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.


