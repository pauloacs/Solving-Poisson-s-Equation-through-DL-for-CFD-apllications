# Solving-Poisson-s-Equation-through-DL-for-CFD-apllications

This repository contains the codes to allow reproduce results from the **Solving Poisson's Equation $
The work is divided with respect to the dissertation chapters as follows:

## - **Chapter 3** - Preliminary works
- 
  - Data-driven methods

  - Physics-informed Neural Networks (PINN)

    - PINN 1
    - PINN 2

    PINN 1 and 2 are based on the Navier-Stokes equations.

    - PINN 3
    - PINN 4

    PINN 3 and 4 are based on the Cauchy-momentum equations and constitutive equations for Newtonian fluid.

![alt text](https://github.com/pauloacs/Solving-Poisson-s-Equation-through-DL-for-CFD-apllications/blob/main/Chapter3/plot_pressure.png)

## - **Chapter 4** - Surrogate model for CFD pressure solver

  - CNN Neural networks (NN)
    - Full CNN architecture

    - PCA based encoder + CNN
    - PCA based encoder + CNN + PCA based decoder

  - MLP NN (with PCA based encoder and decoder)

	Codes for:
	- Training 
	- Predict and evaluate
	Result examples

## - **Chapter 5** - DLPoissonFoam solver

  - Algorithm 1

  - Algorithm 2

## - Generate blockMeshDict

Python scripts for generating blockMeshDict files to use with blockMesh utily from OpenFOAM.
