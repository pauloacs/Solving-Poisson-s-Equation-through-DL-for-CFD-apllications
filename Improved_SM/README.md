# Solving-Poisson-Equation-with-DL-pt2

This repository is a continuation of the work in [Solving-Poisson-s-Equation-through-DL-for-CFD-apllications](https://github.com/pauloacs/Solving-Poisson-s-Equation-through-DL-for-CFD-apllications).

This repository contains two new variants of the Surrogate Models developed in the above project.

## deltaU_to_deltaP Surrogate Model

- **Inputs:** $[U(t) - U(t-1)]$ & $SDF$
- **Output:** $[p(t) - p(t-1)]$

Here a new SM is being developed to drastically improve the accuracy of the Surrogate model for a given Reynolds number.

### Usage

To use it:

1. Enter the directory and install the package
   ```bash
   cd deltaU_to_deltaP/
   pip install .

2. To use it check:
    ```bash
    train_script -h
    evaluate_script -h

### Create your own Surrogate Model

To actually create you own surrogate model follow the description of each entry point and:

#### 1st - **Train your SM** with the train_script entry point

Example:
```bash
train_script --dataset_path dataset_plate.hdf5 --outarray_fn ../blocks_dataset/outarray.h5 --num_sims 5 --num_epoch 5000 --lr 1e-6 --n_samples 1e4 --var_p 0.95 --var_in 0.95 --model_size small --dropout_rate 0.2 --num_ts 5
```

#### 2nd - **Evaluate your SM** with the evaluate_script entry point

Example:
```bash
evaluation_script --model_name model.h5 --var_p 0.95 --var_in 0.95 --dataset_path dataset_plate.hdf5 --n_sims 5 --n_ts 5 --save_plots --show_plots
```

**Enter folder deltaU_to_deltaP for more details**

## U_to_gradP Surrogate Model

- **Inputs:** $U$ & $SDF$
- **Output:** $\nabla p$

Here a different method is being developed to improve the generalization capacity for different Re numbers.

(This SM still needs to be turned into a python package, for now only scripts are available)