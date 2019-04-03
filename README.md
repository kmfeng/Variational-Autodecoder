Implementation of [Variational Auto-decoder](https://arxiv.org/pdf/1903.00840.pdf)
(A. Zadeh, Y.C. Lim, P. Liang, L.-P. Morency, 2019.), by Yao Chong Lim. 

# Introduction

Variational Auto-Decoder refers to **encoderless** implementation of the Auto-Encoding Variational Bayes (AEVB) Algorithm. As opposed to using an encoder to infer the parameters of the posterior of the latent space ![equation](https://latex.codecogs.com/gif.latex?q%28z%7Cx_i%29), (first)  the decoder ![equation](https://latex.codecogs.com/gif.latex?%5Cmathcal%7BF%7D%20%28%5Ccdot%3B%5Ctheta%29) is sampled from using Markov Chain Monte Carlo approaches (MCMC), (second) an arbitrary distribution (which is easy to sample from) is fit using Expectation-Maximization (EM) (third) this distribution is used to sample the neighborhood of a datapoint ![equation](https://latex.codecogs.com/gif.latex?x_i). 

# Requirements
- Python 3

See `requirements_vad.txt` for remaining dependencies. To install,
```
pip install -r requirements_vad.txt
```

# Instructions

## Obtaining the data
Obtain the data from [here], and extract it to `data/`. Then your directory should look like
```
data/
- mnist/
- fashion-mnist/
... other datasets
```

## Generating hyperparameter configurations

For convenient grid search over hyperparameters, use the script at `configs/make_configs.py` to generate
a folder of JSON files representing different hyperparameter configurations.

Usage:
```
python configs/make_configs.py <config name>
```

## Training a model

- `pytorch/train_model.py`

Usage:
```
python pytorch/train_model.py train <dataset> <config name> <config number>
```

Optional arguments can be found using `python pytorch/train_model.py --help`.

## Test on missing

To test a trained model on partial data, use the `test_missing` option. `--missing_mode` must be selected, with either the `fixed` or `random` option.

Example:
```
python pytorch/train_model.py test_missing mnist test_configs 0 --model vae --cuda --log_var -14 --n_test_epochs 500 --test_batch_size 256 --missing_mode fixed
```

## Test on clean

To test a trained model on clean data, use the `test_clean` option.

Example:
```
python pytorch/train_model.py test_clean mnist test_configs 0 --model vae --log_var -14 --cuda --n_test_epochs 5 --test_batch_size 256
```

## Sample commands

```
python configs/make_configs.py test_configs
python pytorch/train_model.py train mnist test_configs 0 --model vae --cuda --log_var -14 --n_train_epochs 500 --n_test_epochs 500 --test_batch_size 256 --batch_size 32
python pytorch/train_model.py test_missing mnist test_configs 0 --model vae --cuda --log_var -14 --n_test_epochs 500 --test_batch_size 256 --missing_mode fixed
```

For the synthetic datasets, you will need to provide the file prefix for the data files:
```
python pytorch/train_model.py train artificial art_vae 800 --cuda --model vae --log_var -14 --dataset_path data/artificial/1/activated_masked_50k
```

## Helper scripts to gather results

```
# gather all results into a single csv file
python scripts/pull_pytorch_errors.py <config name>
python scripts/pull_pytorch_errors.py test_config

# plot predictions and input for mnist-type data
python scripts/plot_mnist.py <mnist / fashion_mnist> <config name> <config number>
python scripts/plot_mnist.py fashion_mnist test_config 0
```
