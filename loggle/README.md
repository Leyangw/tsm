# Loggle Comparison Results

This repository contains code generating comparison results for `loggle`, **SparTSM**. Although `loggle` is available on CRAN as an R package, we run `loggle` from Python for consistency and convenience in our experiments.

## Installation

### Install R and Required Packages

First, ensure that you have R installed. We recommend using **R version 4.0.1**, with corresponding version of RTools(i.e. R version 4.0.1 corresponding to RTools 4.0)

Install the necessary R packages by running the following commands in R:

```r
install.packages("devtools")
install.packages(c("Matrix", "doParallel", "igraph", "glasso", "sm"))
```

### Install Loggle

Install the `loggle` package from GitHub:

```r
library(devtools)
install_github(repo="jlyang1990/loggle")
```

Make sure that you can run `loggle` and `loggle.cv` in R without errors.

### Configure R Environment for Python
To call R from Python, you need to install ```r2py```. Ensure that `PATH`, `R_HOME` environment variables are set correctly and a version of Microsoft Visual C++ Build Tools is installed. 


## Contents of `test_loggle`
The `test_loggle` directory contains the following files for comparison tests related to `loggle`:

- **`Loggle_SparTSM_sine.py`**: Runs a comparison between `loggle` and `SparTSM` using $t$-varying sine function. It generates a figure of $\partial_t \Theta(t)$ estimated by SparTSM and $\Theta(t)$ estimated by 'loggle'.

- **`Loggle_power.py`**: Calculates the power of `loggle` for detecting the linear change for a specific edge in Gaussian Graphical Models. The detection is based on a linear regression, as explained in the paper. 

- **`Loggle_ROC.py`**: Computes the true positive rates and false positive rates over 20 fixed thresholds ranging from 0 to 0.7 for detecting changes in precision matrix, to generate ROC curves.

- **`Loggle_type_i_deterministic.py`** and **`Loggle_type_i_random.py`**: Calculate the Type I error for deterministic and random Gaussian Graphical Models under the null hypothesis of no change, using a heuristic threshold obtained by permutation test. 

## Hyperparameter Selection

Several hyperparameters need to be selected, both for the permutation test and within `loggle` itself.

### Parameter Choices in `loggle`

`loggle` provides `loggle.cv` for selecting optimal parameters `lambda`, `h`, and `d` via cross-validation. However, running cross-validation during permutation tests is computationally expensive. Therefore, we first determine suitable parameters for our settings using `loggle.cv`, and then fix these parameters during the permutation tests.

The chosen parameters are:

- **For `Loggle_SparTSM_sine.py`**:
  - Bandwidth `h`: **0.1**
  - Time step size `d`: **0.01**
  - Regularization parameter `lambda`: **0.03**

- **For `Loggle_type_i_deterministic.py`, `Loggle_type_i_random.py`, `Loggle_ROC.py`, and `Loggle_power.py`**:
  - Bandwidth `h`: **0.2**
  - Time step size `d`: **0.2**
  - Regularization parameter `lambda`: **0.15**


### Permutation Tests

The number of permutations in the permutation test is an important hyperparameter. A low number of permutations can lead to a poor quantile/threshold estimate. We use **100 permutations** in our tests. Note that the process of permutaiton test using `loggle` is time consuming, the result from smaller permutation number might be more acceptable i.e.**50**.
