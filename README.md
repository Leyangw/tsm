# High-Dimensional Differential Parameter Inference in Exponential Family using Time Score Matching (AISTATS 2025)

[![Paper](https://img.shields.io/badge/paper-arxiv.2410.10637-B31B1B.svg)](https://arxiv.org/abs/2410.10637)

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

To create the Conda environment with all the required dependencies, run:

```bash
conda env create -f environment.yaml
```

This command will read the `environment.yaml` file in the repository, which contains all the necessary package information.

### Activate the Environment

After creating the environment, you can activate it by running:

```bash
conda activate tsm
```


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


