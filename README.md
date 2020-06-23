# Regularized Regression Coresets

This project contains the code for the experiments performed in the paper "On Coresets For Regularized Regression" by Rachit Chhaya, Supratim Shit and Anirban Dasgupta accepted at ICML 2020. Here is the link to the arxiv version: https://arxiv.org/abs/2006.05440
There are 3 files containing matlab code for the experiments. 
RLAD_icml.m contains the code for the Regularized Least Absolute Deviation problem on synthetic data. The modifiedlasso_icml.m contains the code for modified lasso experiments and also for the sparsity checking experiment of the modified lasso. Finally the modified lasso experiments on real data are done usibg the code modlassoreal data.m .
We used the following real data:
https://archive.ics.uci.edu/ml/datasets/Combined+Cycle+Power+Plant 
All the files have comments that explain the variables and also what small changes to make to run the codes. We have provided code that will generate new synthetic data at every run. However you should save the data once generated as .mat file and then use it repeatedly.
