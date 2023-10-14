# Solving-Poisson-s-Equation-through-DL-for-CFD-apllications

This repository contains the codes to reproduce the results from **Solving Poisson's Equation through DL for CFD applications** Master's thesis.

The dissertation can be found at FEUP repository: https://sigarra.up.pt/feup/en/pub_geral.pub_view?pi_pub_base_id=547360


The work is divided with respect to the dissertation chapters as follows:

## - **Chapter 3** - Preliminary works
- 
  - Data-driven methods

	|                 Results example                 |
	| ---------------------------------------------------------- |
	| <p align="center"><img src="Chapter3/animations/p_movie.gif" width="380"></p> |

  - Physics-informed Neural Networks (PINN)

    - PINN 1
    - PINN 2

    PINN 1 and 2 are based on the Navier-Stokes equations.

    - PINN 3
    - PINN 4

    PINN 3 and 4 are based on the Cauchy-momentum equations and constitutive equations for Newtonian fluid.

	|                 Results validation               |
	| ---------------------------------------------------------- |
	| <p align="center"><img src="Chapter3/plot_pressure.png" width="380"></p> |

## - **Chapter 4** - Surrogate model for CFD pressure solver

  - CNN Neural networks (NN)
    - Full CNN architecture

    - PCA based encoder + CNN
    - PCA based encoder + CNN + PCA based decoder

  - MLP NN (with PCA based encoder and decoder)
    
	|                  Reconstruction algorithm                  |
	| ---------------------------------------------------------- |
	| <p align="center"><img src="Chapter4/animations/reconstruction.gif" width="380" alt="Reconstruction algorithm"></p> |
	
	  - Result examples (check additonal examples in Chapter4/animations)
	
	| Circular obstacle                         | Rectangular obstacle                        |
	| ------------------------------ | ------------------------------ |
	| ![GIF 1](Chapter4/animations/cil0.gif) | ![GIF 2](Chapter4/animations/rect0.gif) |
	| Triangular obstacle                         | Inclined plate obstacle                       |
	| ![GIF 3](Chapter4/animations/tria0.gif) | ![GIF 4](Chapter4/animations/placa0.gif) |

	Code available for:
	- Training 
	- Prediction and validation
 
## - **Chapter 5** - DLPoissonFoam solver

  - Algorithm 1

  - Algorithm 2

  - Added parallelized implementation

  ### Docker
  
  To allow reproducibility a docker container with the solver is available here. Ensuring you already have the docker installed the container can be pulled with
  
  ```sh
  $ docker pull pauloacs/dlpoissonfoam:latest
  ```
  
  Using 
  
  ```sh
  $ docker run -it  dlpoissonfoam/ofv8:latest bash
  $ source /opt/conda/bin/activate
  $ conda activate python39
  ```
  will create a docker container and run it in a shell. In `/home/foam` a file with the solver and a test case is available. 
  
## - Generate blockMeshDict

Python scripts for generating blockMeshDict files to use with blockMesh utily from OpenFOAM.

