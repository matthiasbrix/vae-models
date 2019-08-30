# Temporal models in Variational Autoencoders

This repository is an implementation of methods in the frame work of variational autoencoders (VAEs). It has implementations of 

1. Vanilla VAE.
2. Conditional VAE.
3. A temporal model that can predict images at a timestamp `t+1` given an image a at timestamp `t` (denoted as TDCVAE).
4. Like 3 but for lung scans (denoted as TDCVAE2).

# Getting started

## Installing

Install the packages/libraries that are enlisted in the `requirements.txt` file. You can also install them with the commands listed below:

```
pip install -r requirements.txt
```
Or if chosing an environment in Anaconda:
```
conda install --file requirements.txt
```

## Data sets

In order to run the model `TDCVAE2`, a dataset of lung scans is required, but this data set is not published. In case it is available, place it in `data/lungscans`. Other data sets (MNIST/LFW) will be downloaded automatically.

# Running the repository

## Training a model

Talk also about that model parameters are centrally modified in the modelparams file. See in the file also the allowed data sets of each model.

Train models either by the jupyter notebooks or the command line. The arguments are below:
```
python solver.py --model <model name> --dataset <data set> [--save_files] [--save_model_state] [--scales] [--thetas]
```
where `--thetas` and/or `--scales` can only be invoked by using model argument `tdcvae` and data set argument `mnist`.
For more help for training, retieve the information about arguments with:
```
python solver.py --help
```

## Producing the plots

After having trained a model, the model can be loaded in the corresponding jupyter notebook. The plots can then be produced in the notebook. **Plots cannot be produced from the script**.