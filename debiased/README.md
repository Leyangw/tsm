# Reproducing plot Figure 3 and 4

## Set-up

We recommend creating a virtual environment with something such as anaconda, with Python version 3.11.4, e.g. in bash
```
conda create -n SparTSM python=3.11.4 ipython
conda activate SparTSM
```
and installing required packages given with the `requirements.txt` file
```
pip install -r requirements.txt
``````

## Reproducing Figure 3
To reproduce coverage experiments, first go to debiased folder
```
cd tsm/src/debiased
```
Modify experiments type to reproduce experiments in ```main.py```, e.g. ```rand inv cov```(defult). Then run
```
python main.py
```
Figure 3 can be found in the folder 'logs/'.

The hyperparameters are set as described in the appendix of submitted manucript, both $\lambda_{lasso}$ and $\lambda_{1,2}$ are set to be $\sqrt{\frac{2log k}{n}}$. 

## Reproducing Figure 4
To reproduce the points of power plot, user needs to manually change the value $\Theta_{1,2}^\prime (t) = 1,...,10$ in line 46-47 or 70-71
```
inv_cov[0,1] =
inv_cov[1,0] = 
```
Fill the empty space with the value of interest.

Loggle results are independently generated by running commands provided in ```../loggle/README.md```. 

### Note: 
Different seeds may result in different results, but generally the results do not vary much.
