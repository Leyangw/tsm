# High-Dimensional Differential Parameter Inference in Exponential Family using Time Score Matching (AISTATS 2025)

[![Paper](https://img.shields.io/badge/paper-arxiv.2410.10637-B31B1B.svg)](https://arxiv.org/abs/2410.10637)
[![AISTATS 2025](https://img.shields.io/badge/paper-PMLR.v258-0033CC.svg)](https://proceedings.mlr.press/v258/williams25a.html)

Official Python implementation of the paper [High-Dimensional Differential Parameter Inference in Exponential Family using Time Score Matching](https://arxiv.org/abs/2410.10637), published at AISTATS 2025.

In this paper, we introduce a method that tackle the differential parameter estimation in exponential family in a continuous setting.  The main idea is treating the time score function of an exponential family model as a linear model of the differential parameter for direct estimation. We use time score matching to estimate parameter derivatives.

## Prerequisites

Before you begin, ensure you have met the following requirements:
* You have installed the latest version of [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html).
* You have a `Windows/Linux/Mac` machine.

## Installation

To install the necessary packages and set up the environment, follow these steps:

### Clone the Repository

First, clone the repository to your local machine:

```bash
git clone https://github.com/Leyangw/tsm.git
cd tsm
```

### Create the Conda Environment

We recommend creating a virtual environment with something such as anaconda, with Python version 3.11.4, e.g. in bash

```bash
conda create -n tsm python=3.11.4 ipython
conda activate tsm
```

and installing required packages given with the `requirements.txt` file
```
pip install -r requirements.txt
``````
to ensure every package is installed correctly for this repo. 


## Current Folder

Code for reproducing SparTSM results:

- ```demo_fourier.ipynb```: Figure 1
- ```demo_ROC.ipynb```, ```demo_ROC_ratio.py```: Figure 2
- ```demo_senate.ipynb```: Figure 5

Requires python 3.10+ with ```torch```, ```panda```, ```networkx```, and ```matplotlib``` packages.  

## Folder Structure

Folder for reproducing debiased estimator (SparTSM+) results (Figure 3, 4)
```
debiased/
```
Please see README file in the folder for detailed procedures for reproducing the results. 


Folder for reproducing Loggle method reuslts
```
loggle/
```
Note that Loggle method was run and results were collected separately from the rest methods as it requires a special R environment. 

## Citation

If you find our paper, code, and/or data useful for your research, please cite our paper:

```
@inproceedings{williams2025high,
  title={High-Dimensional Differential Parameter Inference in Exponential Family using Time Score Matching},
  author={Williams, Daniel J and Wang, Leyang and Ying, Qizhen and Liu, Song and Kolar, Mladen},
  booktitle={International Conference on Artificial Intelligence and Statistics},
  year={2025}
}
```



