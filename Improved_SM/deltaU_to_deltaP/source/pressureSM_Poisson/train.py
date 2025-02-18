import os

# Set environment variables for reproducibility **before** any other imports
os.environ['PYTHONHASHSEED'] = '0'
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import random
import numpy as np
import tensorflow as tf

# Limit TensorFlow threading parallelism
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

# Set seeds for reproducibility
random.seed(0)
np.random.seed(0)
tf.keras.utils.set_random_seed(0)
tf.random.set_seed(0)

# Optional: Enable GPU memory growth if using GPUs (uncomment if needed)
# physical_devices = tf.config.list_physical_devices('GPU')
# for device in physical_devices:
#     tf.config.experimental.set_memory_growth(device, True)

# Dask configuration for reproducibility
import dask
dask.config.set(scheduler='threads')

# Standard library imports
import shutil
import time
import math
import itertools
from ctypes import py_object
import pickle as pk

# Third-party library imports
import tables
import h5py
import matplotlib
import matplotlib.pyplot as plt
import scipy.spatial.qhull as qhull
from scipy.spatial import cKDTree as KDTree, distance
import matplotlib.path as mpltPath
from shapely.geometry import MultiPoint
from sklearn.decomposition import PCA, IncrementalPCA

# Additional scientific computing and data processing libraries
from pyDOE import lhs
from numba import njit
import dask_ml
import dask_ml.preprocessing
import dask_ml.decomposition


def smart_arcsin_smooth_transform(field, central_dist_std_multiplier, scale_output=False, new_min=0, new_max=1):
    """
    Apply a smart smoothing transformation that scales the input field based on mean and standard deviation,
    then applies the arcsin transformation.

    Parameters:
    - field (array-like): Input spatial field (e.g., a 2D array or list of values).
    - scale_output (bool): Whether to scale the transformed field back to a specific range.
    - new_min (float): Minimum of the new range (if scaling is applied).
    - new_max (float): Maximum of the new range (if scaling is applied).

    Returns:
    - transformed_field (array-like): Arcsin-transformed and optionally scaled field.
    """
    # Step 1: Calculate mean and standard deviation
    mean = np.mean(field)
    std = np.std(field)

    # Step 2: Define central range [mean - 2*std, mean + 2*std]
    lower_bound = mean - central_dist_std_multiplier * std
    upper_bound = mean + central_dist_std_multiplier * std

    # Step 3: Scale the entire field such that:
    # values in [lower_bound, upper_bound] are scaled to [-1, 1]
    # values < lower_bound are < -1
    # values > upper_bound are > 1
    def scale_to_range(x):
        if x < lower_bound:
            return -1 - (x - lower_bound) /lower_bound  # Outlier below lower bound
        elif x > upper_bound:
            return 1 + (x - upper_bound) / upper_bound  # Outlier above upper bound
        else:
            return 2 * (x - lower_bound) / (upper_bound - lower_bound) - 1  # Central values

    scaled_field = np.vectorize(scale_to_range)(field)

    # Step 4: Apply arcsin transformation
    #arcsin_field = np.arcsin(scaled_field) #np.arcsin(np.clip((scaled_field + 1) / 2, 0, 1))  # Scale to [0, 1] for arcsin
    arcsin_field = np.arcsinh(scaled_field)

    # Optional Step 5: Scale the transformed field back to [new_min, new_max] range if requested
    if scale_output:
        min_arcsin = np.min(arcsin_field)
        max_arcsin = np.max(arcsin_field)
        scaled_arcsin_field = (arcsin_field - min_arcsin) / (max_arcsin - min_arcsin) * (new_max - new_min) + new_min
        return scaled_arcsin_field

    return arcsin_field


def num_derivative(var, dim):
    """
    Computes the derivative d(var)/d(dim) using central differences.
    
    Parameters:
    dim (numpy array): The independent variable (e.g., x or y).
    var (numpy array): The dependent variable (e.g., the interest variable).
    
    Returns:
    deriv (numpy array): The numerical derivative d(var)/d(dim).
    """
    # Ensure the inputs are numpy arrays
    dim = np.asarray(dim)
    var = np.asarray(var)
    
    # Check that dim and var have the same length
    if len(dim) != len(var):
        raise ValueError("dim and var must have the same length.")
    
    # Compute the central difference for the interior points
    deriv = np.zeros_like(var)
    
    deriv[1:-1] = (var[2:] - var[:-2]) / (dim[2:] - dim[:-2])  # central difference

    # For the boundary points, use forward and backward differences
    deriv[0] = (var[1] - var[0]) / (dim[1] - dim[0])  # forward difference
    deriv[-1] = (var[-1] - var[-2]) / (dim[-1] - dim[-2])  # backward difference
    
    return deriv

def compute_rbf_derivatives(x, y, z):
    """
    Compute the partial derivatives dz/dx and dz/dy using RBF interpolation.

    Parameters:
    x, y, z (numpy arrays): Arrays representing the 2D coordinates (x, y) and the function value z.

    Returns:
    dz_dx, dz_dy (numpy arrays): Arrays of partial derivatives with respect to x and y.
    """
    # Create RBF interpolator for the data
    rbf = Rbf(x, y, z, function='multiquadric', epsilon=2)  # You can choose different functions like 'linear', 'cubic', etc.

    # Define derivative functions by differentiating the RBF
    dz_dx = rbf(x, y, dx=1, dy=0)  # Partial derivative with respect to x
    dz_dy = rbf(x, y, dx=0, dy=1)  # Partial derivative with respect to y

    return dz_dx, dz_dy

def compute_delaunay_derivatives(x, y, z, tri):
    """
    Compute approximate derivatives dz/dx and dz/dy using Delaunay triangulation.
    
    Parameters:
    x, y, z (numpy arrays): Arrays representing the 2D coordinates (x, y) and the function value z.

    Returns:
    grad_x, grad_y (numpy arrays): Approximate derivatives of z with respect to x and y.
    """

    # Initialize arrays to store gradients
    dz_dx = np.zeros(len(z))
    dz_dy = np.zeros(len(z))

    # Loop through triangles and estimate gradients
    for simplex in tri.simplices:
        x_triangle = x[simplex]
        y_triangle = y[simplex]
        z_triangle = z[simplex]

        # Create the matrix A and solve A * [dz_dx, dz_dy] = dz for this triangle
        A = np.vstack([x_triangle - x_triangle[0], y_triangle - y_triangle[0]]).T
        dz = z_triangle - z_triangle[0]

        # Solve for the gradient vector [dz_dx, dz_dy]
        grad = np.linalg.lstsq(A, dz, rcond=None)[0]
        dz_dx[simplex] = grad[0]  # Assign gradient estimate for dz/dx
        dz_dy[simplex] = grad[1]  # Assign gradient estimate for dz/dy

    return dz_dx, dz_dy

class Training:

  def __init__(self, delta, block_size, var_p, var_in, hdf5_paths, n_samples, num_sims, first_t, last_t, standardization_method, n_chunks, k):
    self.delta = delta
    self.block_size = block_size
    self.var_in = var_in
    self.var_p = var_p
    self.paths = hdf5_paths
    self.n_samples = n_samples
    self.num_sims = num_sims[0]
    self.first_t = first_t
    self.last_t = last_t
    self.standardization_method = standardization_method
    self.n_chunks = n_chunks
    self.k = k

  #@njit     ### with numba.njit is much faster but is returning an error when using it within the class ###
  def index(self, array, item):
    for idx, val in np.ndenumerate(array):
        if val == item:
            return idx
    # If no item was found return None, other return types might be a problem due to
    # numbas type inference.

  def interp_weights(self, xyz, uvw):
    d = 2 #2d interpolation
    tri = qhull.Delaunay(xyz)
    simplex = tri.find_simplex(uvw)
    vertices = np.take(tri.simplices, simplex, axis=0)
    temp = np.take(tri.transform, simplex, axis=0)
    delta = uvw - temp[:, d]
    bary = np.einsum('njk,nk->nj', temp[:, :d, :], delta)
    return vertices, np.hstack((bary, 1 - bary.sum(axis=1, keepdims=True)))

  def interpolate(self, values, vtx, wts):
    return np.einsum('nj,nj->n', np.take(values, vtx), wts)

  def interpolate_fill(self, values, vtx, wts, fill_value=np.nan):  #this would be the function to fill with nan 
    ret = np.einsum('nj,nj->n', np.take(values, vtx), wts)  #does not work yet
    ret[np.any(wts < 0, axis=1)] = fill_value
    return ret
  
  def create_uniform_grid(self, x_min, x_max,y_min,y_max): 
    """
    Creates a uniform quadrangular grid envolving every cell of the mesh
    """
    X0 = np.linspace(x_min + self.delta/2 , x_max - self.delta/2 , num = int(round( (x_max - x_min)/self.delta )) )
    Y0 = np.linspace(y_min + self.delta/2 , y_max - self.delta/2 , num = int(round( (y_max - y_min)/self.delta )) )

    XX0, YY0 = np.meshgrid(X0,Y0)
    return XX0.flatten(), YY0.flatten()

  
  ################################################################################
  ## Neural Networks architectures
  ################################################################################

  def densePCA(self, n_layers, depth=512, dropout_rate=None, regularization=None):
      """
      Creates the MLP NN.
      """
      
      # Explicitly initialize the NN weights to ensure reproducibility
      initializer = tf.keras.initializers.RandomUniform(minval=-0.05, maxval=0.05, seed=0)

      # Input layer
      inputs = tf.keras.layers.Input(shape=(int(self.PC_input),))
      
      # Adjust depth if a single value is provided
      if isinstance(depth, int):
          depth = [depth] * n_layers
      
      # Set up regularization if provided
      if regularization is not None:
          regularizer = tf.keras.regularizers.l2(regularization)
          print(f'\nUsing L2 regularization. Value: {regularization}\n')
      else:
          regularizer = None

      # Build the first Dense layer with initializer and optional regularization
      x = tf.keras.layers.Dense(
          depth[0], 
          activation='relu', 
          kernel_initializer=initializer, 
          kernel_regularizer=regularizer
      )(inputs)
      
      # Add dropout if specified
      if dropout_rate is not None:
          x = tf.keras.layers.Dropout(dropout_rate)(x)

      # Build the remaining layers
      for i in range(1, n_layers):
          x = tf.keras.layers.Dense(
              depth[i], 
              activation='relu', 
              kernel_initializer=initializer, 
              kernel_regularizer=regularizer
          )(x)
          if dropout_rate is not None:
              x = tf.keras.layers.Dropout(dropout_rate)(x)
      
      # Output layer (no activation)
      outputs = tf.keras.layers.Dense(self.PC_p, kernel_initializer=initializer)(x)

      # Create the model
      model = tf.keras.Model(inputs, outputs, name="MLP")
      
      # Print the model summary
      model.summary()
      
      return model
      
  def densePCA_attention(self, n_layers=3, depth=[512], dropout_rate=None, regularization=None):
    """
    Creates the MLP with an attention mechanism.
    """
    inputs = tf.keras.kayers.Input((int(self.PC_input),))
    if len(depth) == 1:
        depth = [depth[0]] * n_layers
    
    # Regularization parameter
    regularizer = tf.keras.regularizers.l2(regularization) if regularization else None
    
    x = tf.keras.layers.Dense(depth[0], activation='relu', kernel_regularizer=regularizer)(inputs)
    if dropout_rate is not None:
        x = tf.keras.layers.Dropout(dropout_rate)(x)
    
    # Applying a multi-head attention layer
    x = tf.expand_dims(x, 1)  # Add a new dimension for the sequence length
    attn_output = tf.keras.layers.MultiHeadAttention(num_heads=8, key_dim=64)(x, x)
    attn_output = tf.keras.layers.LayerNormalization()(attn_output)
    attn_output = tf.squeeze(attn_output, 1)  # Remove the added dimension
    
    # Adding additional dense layers
    for i in range(1, n_layers):
        x = tf.keras.layers.Dense(depth[i], activation='relu', kernel_regularizer=regularizer)(attn_output)
        if dropout_rate is not None:
            x = tf.keras.layers.Dropout(dropout_rate)(x)
        attn_output = tf.keras.layers.LayerNormalization()(x + attn_output)  # Residual connection
    
    outputs = tf.keras.layers.Dense(self.PC_p)(attn_output)

    model = tf.keras.Model(inputs, outputs, name="MLP_with_Attention")
    print(model.summary())

    return model

  def conv1D_PCA(self, n_layers=3, depth=[512], dropout_rate=None, regularization=None, kernel_size=3):
      """
      Creates a 1D ConvNet with regularization and dropout, similar to an MLP.
      """
      
      # Define input layer
      inputs = tf.keras.layers.Input(shape=(self.PC_input, 1))  # 1D Conv input shape requires an extra dimension
      
      if len(depth) == 1:
          depth = [depth[0]] * n_layers
      
      # Regularization parameter
      regularizer = tf.keras.regularizers.l2(regularization) if regularization else None
      
      # First convolutional layer
      x = tf.keras.layers.Conv1D(
          filters=depth[0], 
          kernel_size=kernel_size, 
          activation='relu',
          padding='same',
          kernel_regularizer=regularizer
      )(inputs)

      # Optional dropout
      if dropout_rate:
          x = tf.keras.layers.Dropout(dropout_rate)(x)
      
      # Additional convolutional layers
      for i in range(1, n_layers):
          x = tf.keras.layers.Conv1D(
              filters=depth[i], 
              kernel_size=kernel_size,
              padding='same',
              activation='relu', 
              kernel_regularizer=regularizer
          )(x)
          
          if dropout_rate:
              x = tf.keras.layers.Dropout(dropout_rate)(x)
      
      # Flatten and final dense layer
      x = tf.keras.layers.Flatten()(x)  # Convert 1D convolution output to a 1D vector
      outputs = tf.keras.layers.Dense(self.PC_p)(x)

      # Create and compile the model
      model = tf.keras.Model(inputs, outputs, name="1D_ConvNet")

      print(model.summary())

      return model

  #########################################################################
  ## 3 options currently available to work with PC data
  #########################################################################
  
  def domain_dist(self, i,top_boundary, obst_boundary, xy0):

    # boundaries indice
    indice_top = self.index(top_boundary[i,0,:,0] , -100.0 )[0]
    top = top_boundary[i,0,:indice_top,:]
    max_x, max_y, min_x, min_y = np.max(top[:,0]), np.max(top[:,1]) , np.min(top[:,0]) , np.min(top[:,1])

    is_inside_domain = ( xy0[:,0] <= max_x)  * ( xy0[:,0] >= min_x ) * ( xy0[:,1] <= max_y ) * ( xy0[:,1] >= min_y )

    # regular polygon for testing

    # # Matplotlib mplPath
    # path = mpltPath.Path(top_inlet_outlet)
    # is_inside_domain = path.contains_points(xy0)
    # print(is_inside_domain.shape)

    indice_obst = self.index(obst_boundary[i,0,:,0] , -100.0 )[0]
    obst = obst_boundary[i,0,:indice_obst,:]

    obst_points =  MultiPoint(obst)

    hull = obst_points.convex_hull       #only works for convex geometries
    hull_pts = hull.exterior.coords.xy    #have a code for any geometry . enven concave https://stackoverflow.com/questions/14263284/create-non-intersecting-polygon-passing-through-all-given-points/47410079
    hull_pts = np.c_[hull_pts[0], hull_pts[1]]

    path = mpltPath.Path(hull_pts)
    is_inside_obst = path.contains_points(xy0)

    domain_bool = is_inside_domain * ~is_inside_obst

    top = top[0:top.shape[0]:5,:]   #if this has too many values, using cdist can crash the memmory since it needs to evaluate the distance between ~1M points with thousands of points of top
    obst = obst[0:obst.shape[0]:5,:]

    #print(top.shape)

    sdf = np.minimum( distance.cdist(xy0,obst).min(axis=1) , distance.cdist(xy0,top).min(axis=1) ) * domain_bool
    #print(np.max(distance.cdist(xy0,top).min(axis=1)))
    #print(np.max(sdf))

    return domain_bool, sdf

  def sample_blocks(self, grid, x_list, obst_list, y_list, N):
    """Sample N blocks from each time step based on LHS"""

    lb = np.array([0 + self.block_size * self.delta/2 , 0 + self.block_size * self.delta/2 ])
    ub = np.array([(self.x_max-self.x_min) - self.block_size * self.delta/2, (self.y_max-self.y_min) - self.block_size * self.delta/2])

    XY = lb + (ub-lb)*lhs(2,N)
    XY_indices = (np.round(XY/self.delta)).astype(int)

    new_XY_indices = [tuple(row) for row in XY_indices]
    XY_indices = np.unique(new_XY_indices, axis=0)
    count=0
    for [jj, ii] in XY_indices:

            i_range = [int(ii - self.block_size/2), int( ii + self.block_size/2) ]
            j_range = [int(jj - self.block_size/2), int( jj + self.block_size/2) ]

            x_u = grid[0, i_range[0]:i_range[1] , j_range[0]:j_range[1] , 0:3 ]
            x_obst = grid[0, i_range[0]:i_range[1] , j_range[0]:j_range[1] , 3:4 ]
            y = grid[0, i_range[0]:i_range[1] , j_range[0]:j_range[1] , 4:5 ]

            # Remove all the blocks with delta_U = 0 and delta_p = 0
            if not ((x_u == 0).all() and (y == 0).all()):
              x_list.append(x_u)
              obst_list.append(x_obst)
              y_list.append(y)
            else:
              count += 1

    print(f'{count} blocks discarded')
    return x_list, obst_list, y_list

  def process_time_step(self, j, data_limited, vert, weights, indices, sdfunct, phi):
    """
    """

    Ux = data_limited[...,0] #values
    Uy = data_limited[...,1] #values

    delta_p = data_limited[...,7] #values
    delta_Ux = data_limited[...,5] #values
    delta_Uy = data_limited[...,6] #values
    p = data_limited[...,2] #values
    p_prev = p - delta_p

    # cell center coordinates
    x = data_limited[...,3]
    y = data_limited[...,4]

    U_max_norm = np.max(np.sqrt(np.square(Ux) + np.square(Uy)))
    deltaU_max_norm = np.max(np.sqrt(np.square(delta_Ux) + np.square(delta_Uy)))
    # Ignore time steps with minimal changes ...
    # there is not point in computing error metrics for these
    # it would exagerate the delta_p errors and give ~0% errors in p
    threshold = 1e-4
    print(deltaU_max_norm)
    print(U_max_norm)
    irrelevant_ts = (deltaU_max_norm/U_max_norm) < threshold or deltaU_max_norm < 1e-6 or U_max_norm < 1e-6

    if irrelevant_ts:
       print(f"\n\n Irrelevant time step, skipping it...")
       self.stationary_ts += 1
       return 0

    self.stationary_ts = 0

    # Representative length scales
    L = phi
    #U = 1
    U = U_max_norm

    # The derivation ca be done from the the data points - very heavy
    # or from the python grid - cheap

    Ux_interp = self.interpolate_fill(Ux, vert, weights)
    Uy_interp = self.interpolate_fill(Uy, vert, weights)
    p_prev_interp = self.interpolate_fill(p_prev, vert, weights)
    delta_Ux_interp = self.interpolate_fill(delta_Ux, vert, weights)
    delta_Uy_interp = self.interpolate_fill(delta_Uy, vert, weights)
    delta_p_interp = self.interpolate_fill(delta_p, vert, weights)

    # 1D vectors to 2D arrays
    ux_grid = np.zeros(shape=(self.grid_shape_y, self.grid_shape_x))
    uy_grid = np.zeros(shape=(self.grid_shape_y, self.grid_shape_x))
    p_prev_grid = np.zeros(shape=(self.grid_shape_y, self.grid_shape_x))
    delta_ux_grid = np.zeros(shape=(self.grid_shape_y, self.grid_shape_x))
    delta_uy_grid = np.zeros(shape=(self.grid_shape_y, self.grid_shape_x))
    #
    ux_grid[tuple(indices.T)] = Ux_interp.reshape(Ux_interp.shape[0])
    uy_grid[tuple(indices.T)] = Uy_interp.reshape(Uy_interp.shape[0])
    p_prev_grid[tuple(indices.T)] = p_prev_interp.reshape(p_prev_interp.shape[0])
    delta_ux_grid[tuple(indices.T)] = delta_Ux_interp.reshape(delta_Ux_interp.shape[0])
    delta_uy_grid[tuple(indices.T)] = delta_Uy_interp.reshape(delta_Uy_interp.shape[0])

    def gradient_with_nan_direct_neighbors(grid):
        # Compute gradients using np.gradient
        grad_y, grad_x = np.gradient(grid)

        # Create a mask of where the grid has NaN values
        nan_mask = np.isnan(grid)

        # Loop over the grid to check only direct neighbors (up, down, left, right)
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
              # Check the four direct neighbors (up, down, left, right)
              if (i > 0 and nan_mask[i-1, j]) or (i < grid.shape[0] - 1 and nan_mask[i+1, j]) or \
                (j > 0 and nan_mask[i, j-1]) or (j < grid.shape[1] - 1 and nan_mask[i, j+1]):
                    grad_x[i, j] = np.nan
                    grad_y[i, j] = np.nan
              if nan_mask[i,j]:
                grad_x[i, j] = np.nan
                grad_y[i, j] = np.nan

        return grad_y, grad_x

    p_prev_grid[sdfunct[...,0]==0] = np.nan
    ux_grid[sdfunct[...,0]==0] = np.nan
    uy_grid[sdfunct[...,0]==0] = np.nan

    dUx_dy, dUx_dx = gradient_with_nan_direct_neighbors(ux_grid)
    dUy_dy, dUy_dx = gradient_with_nan_direct_neighbors(uy_grid)

    dUx_dy[np.isnan(dUx_dy)] = 0
    dUx_dx[np.isnan(dUx_dx)] = 0
    dUy_dy[np.isnan(dUy_dy)] = 0
    dUy_dx[np.isnan(dUy_dx)] = 0
      
    # Defining the inputs:
    Poisson_term_1 = (dUx_dx * dUx_dx + 2 * dUx_dy * dUy_dx + dUy_dy * dUy_dy) * L**2/U**2
    #Poisson_term_1 = ndimage.gaussian_filter(Poisson_term_1, sigma=(2, 2), order=0)
    #Poisson_term_2 = (d2p_prev_dx2 + d2p_prev_dy2) * L**2/U**2
    #Poisson_term_2 = ndimage.gaussian_filter(Poisson_term_2, sigma=(2, 2), order=0)
    #res_Poisson = - Poisson_term_1 - Poisson_term_2
    Poisson_term_1_before = Poisson_term_1

    # Gradient field have very high variance, with extremely high values near the obstacle
    # Which ruins the SM model ability to learn.
    # Therefore we apply a arcsin transformation to those with the intent of smoothing the larger values of the field

    # Pressure poisson eq residual - not used
    #res_Poisson = ndimage.gaussian_filter(res_Poisson, sigma=(5, 5), order=0)
    #limit = 0.05
    #res_Poisson = smart_arcsin_smooth_transform(res_Poisson, limit)

    # Do not change values in the range [mean - k*std, mean + k*std] - 1st order derivatives
    print(f'k: {self.k}')
    Poisson_term_1 = smart_arcsin_smooth_transform(Poisson_term_1, self.k)



    delta_Ux = delta_ux_grid / U
    delta_Uy = delta_uy_grid / U
    p_prev = p_prev_grid / U**2

    # The output
    delta_p_adim = delta_p_interp/pow(U,2.0) 
    delta_p_grid = np.zeros(shape=(self.grid_shape_y, self.grid_shape_x))
    delta_p_grid[tuple(indices.T)] = delta_p_adim.reshape(delta_p_adim.shape[0])

    # # 1D vectors to 2D arrays
    grid = np.zeros(shape=(1, self.grid_shape_y, self.grid_shape_x, 5))
    grid[0,:,:,0] = Poisson_term_1
    #grid[0,:,:,1] = p_prev
    grid[0,:,:,1] = delta_Ux
    grid[0,:,:,2] = delta_Uy
    grid[0,:,:,3] = sdfunct[...,0]
    grid[0,:,:,4] = delta_p_grid

    debug_plot = False
    if debug_plot:
      fig, axes = plt.subplots(3, 2, figsize=(15, 25))

      titles = [
        "res poisson", "delta_Ux","delta_Uy",
        "SDF", "Output - deltaP"
      ]

      for i in range(len(titles)):
        ax = axes[i//2, i%2]
        im = ax.imshow(grid[0, :, :, i], cmap='viridis', aspect='auto')
        ax.set_title(titles[i])
        fig.colorbar(im, ax=ax)

      plt.tight_layout()
      plt.show()
      plt.close()

    x_list = []
    obst_list = []
    y_list = []
    
    # Setting any nan value to 0
    grid[np.isnan(grid)] = 0

    #How many rotations to do:
    N_rotation = 2
    N = int(self.n_samples/N_rotation/(self.last_t-self.first_t))

    x_list, obst_list, y_list = self.sample_blocks(grid, x_list, obst_list, y_list, N)

    # Rotate and sample
    grid_y_inverted = grid[:, ::-1, :, :]
    x_list, obst_list, y_list = self.sample_blocks(grid_y_inverted, x_list, obst_list, y_list, N)

    # Rotate and sample
    #grid_y_inverted = grid[:, :, ::-1, :]
    #x_list, obst_list, y_list = self.sample_blocks(grid_y_inverted, x_list, obst_list, y_list, N)

    # Rotate and sample
    #grid_y_inverted = grid[:, ::-1, ::-1, :]
    #x_list, obst_list, y_list = self.sample_blocks(grid_y_inverted, x_list, obst_list, y_list, N)

    x_array = np.array(x_list, dtype = 'float32')
    obst_array = np.array(obst_list, dtype = 'float32')
    y_array = np.array(y_list, dtype = 'float32')

    self.max_abs_Poisson_term_1_list.append(np.max(np.abs(x_array[...,0])))
    self.max_abs_delta_ux_list.append(np.max(np.abs(x_array[...,1])))
    self.max_abs_delta_uy_list.append(np.max(np.abs(x_array[...,2])))
    self.max_abs_dist_list.append(np.max(np.abs(obst_array[...,0])))

    # Setting the average pressure in each block to 0
    for step in range(y_array.shape[0]):
      y_array[step,...][obst_array[step,...] != 0] -= np.mean(y_array[step,...][obst_array[step,...] != 0])
    
    self.max_abs_delta_p_list.append(np.max(np.abs(y_array[...,0])))

    array = np.c_[x_array,obst_array,y_array]
    
    # Removing duplicate data
    reshaped_array = array.reshape(array.shape[0], -1)
    # Find unique rows
    unique_indices = np.unique(reshaped_array, axis=0, return_index=True)[1]
    unique_array = array[unique_indices]

    print(f"Writting t{j + self.first_t} to {self.filename}", flush=True)
    file = tables.open_file(self.filename, mode='a')
    file.root.data.append(np.array(unique_array, dtype = 'float16'))
    file.close()

  def process_sim(self, i):
    """
    """
    
    phi_list = np.loadtxt('phis.txt', dtype=float)
    phi = phi_list[i]
    print(f'phi {phi}')
    hdf5_file = h5py.File(self.dataset_path, "r")
    data = np.array(hdf5_file["sim_data"][i:i+1, self.first_t:self.last_t, ...], dtype='float32')
    top_boundary = hdf5_file["top_bound"][i:i+1, self.first_t:self.last_t, ...]
    obst_boundary = hdf5_file["obst_bound"][i:i+1, self.first_t:self.last_t,  ...]
    hdf5_file.close()          

    indice = self.index(data[0,0,:,0] , -100.0 )[0]
    data_limited = data[0,0,:indice,:]

    self.x_min = round(np.min(data_limited[...,3]),2)
    self.x_max = round(np.max(data_limited[...,3]),2)

    self.y_min = round(np.min(data_limited[...,4]),2) #- 0.1
    self.y_max = round(np.max(data_limited[...,4]),2) #+ 0.1


    ######### -------------------- Assuming constant mesh, the following can be done out of the for cycle ------------------------------- ##########

    X0, Y0 = self.create_uniform_grid(self.x_min, self.x_max, self.y_min, self.y_max)
    xy0 = np.concatenate((np.expand_dims(X0, axis=1),np.expand_dims(Y0, axis=1)), axis=-1)
    points = data_limited[...,3:5] #coordinates

    vert, weights = self.interp_weights(points, xy0) #takes ~100% of the interpolation cost and it's only done once for each different mesh/simulation case

    domain_bool, sdf = self.domain_dist(0, top_boundary, obst_boundary, xy0)

    div = 1 #parameter defining the sliding window vertical and horizontal displacements
    
    self.grid_shape_y = int(round((self.y_max-self.y_min)/self.delta)) 
    self.grid_shape_x = int(round((self.x_max-self.x_min)/self.delta)) 

    count_ = data.shape[1]* int(self.grid_shape_y/div - self.block_size/div + 1 ) * int(self.grid_shape_x/div - self.block_size/div + 1 )

    count = 0
    cicle = 0

    #arrange data in array: #this can be put outside the j loop if the mesh is constant 

    x0 = np.min(X0)
    y0 = np.min(Y0)
    dx = self.delta
    dy = self.delta

    indices= np.zeros((X0.shape[0],2))
    obst_bool  = np.zeros((self.grid_shape_y,self.grid_shape_x,1))
    sdfunct = np.zeros((self.grid_shape_y,self.grid_shape_x,1))

    delta_p = data_limited[...,7:8] #values
    p_interp = self.interpolate_fill(delta_p, vert, weights) 
  
    for (step, x_y) in enumerate(xy0):  
        if domain_bool[step] * (~np.isnan(p_interp[step])) :
            jj = int(round((x_y[...,0] - x0) / dx))
            ii = int(round((x_y[...,1] - y0) / dy))

            indices[step,0] = ii
            indices[step,1] = jj
            sdfunct[ii,jj,:] = sdf[step]
            obst_bool[ii,jj,:]  = int(1)

    indices = indices.astype(int)
    
    # Number of subsquent t's with very small variations
    self.stationary_ts = 0
    for j in range(data.shape[1]):  #100 for both data and data_rect

      # go from the last time to the first to access if the simulation is stationary
      #j = (data.shape[1] -1) - j
      data_limited = data[0,j,:indice,:]#[mask_x]
      self.process_time_step(j, data_limited, vert, weights, indices, sdfunct, phi)
      if self.stationary_ts > 5: 
        print('This simulation is stationary, ignoring it...')
        break

  def read_dataset(self):
    """
    """
    #pathCil, pathRect, pathTria , pathPlate = self.paths[0], self.paths[1], self.paths[2], self.paths[3]
    self.dataset_path = self.paths[0]

    NUM_COLUMNS = 5

    file = tables.open_file(self.filename, mode='w')
    atom = tables.Float32Atom()

    array_c = file.create_earray(file.root, 'data', atom, (0, self.block_size, self.block_size, NUM_COLUMNS))
    file.close()

    self.max_abs_Poisson_term_1_list = []
    self.max_abs_delta_ux_list = []
    self.max_abs_delta_uy_list = []
    self.max_abs_dist_list  = []
    self.max_abs_delta_p_list  = []

    for i in range(np.sum(self.num_sims)):
      print(f"\nProcessing sim {i}/{np.sum(self.num_sims)}\n", flush=True)
      self.process_sim(i)

    self.max_abs_Poisson_term_1 = np.max(np.abs(self.max_abs_Poisson_term_1_list))
    self.max_abs_delta_ux = np.max(np.abs(self.max_abs_delta_ux_list))
    self.max_abs_delta_uy = np.max(np.abs(self.max_abs_delta_uy_list))
    self.max_abs_dist = np.max(np.abs(self.max_abs_dist_list))
    self.max_abs_delta_p = np.max(np.abs(self.max_abs_delta_p_list))

    np.savetxt('maxs', [
      self.max_abs_Poisson_term_1, self.max_abs_delta_ux, self.max_abs_delta_uy, 
      self.max_abs_dist, self.max_abs_delta_p
    ])

    return 0

  @tf.function
  def train_step(self, inputs, labels):
    with tf.GradientTape() as tape:
      predictions = self.model(inputs, training=True)
      loss=self.loss_object(labels, predictions)

    gradients = tape.gradient(loss, self.model.trainable_variables)
    self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
    return loss

  #@tf.function
  def perform_validation(self):

    losses = []

    for (x_val, y_val) in self.test_dataset:
      x_val = tf.cast(x_val[...,0,0], dtype='float32')
      y_val = tf.cast(y_val[...,0,0], dtype='float32')

      val_logits = self.model(x_val)
      val_loss = self.loss_object(y_true = y_val , y_pred = val_logits)
      losses.append(val_loss)

    return losses
  
  def my_mse_loss(self):
    def loss_f(y_true, y_pred):

      loss = tf.reduce_mean(tf.square(y_true - y_pred) )

      return 1e6 * loss
    return loss_f

  def _bytes_feature(self, value):
      """Returns a bytes_list from a string / byte."""
      if isinstance(value, type(tf.constant(0))):
          value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
      return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
  def _float_feature(self, value):
      """Returns a float_list from a float / double."""
      return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))
  def _int64_feature(self, value):
      """Returns an int64_list from a bool / enum / int / uint."""
      return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

  def parse_single_image(self, input_parse, output_parse):

    #define the dictionary -- the structure -- of our single example
    data = {
    'height' : self._int64_feature(input_parse.shape[0]),
          'depth_x' : self._int64_feature(input_parse.shape[1]),
          'depth_y' : self._int64_feature(output_parse.shape[1]),
          'raw_input' : self._bytes_feature(tf.io.serialize_tensor(input_parse)),
          'output' : self._bytes_feature(tf.io.serialize_tensor(output_parse)),
      }

    #create an Example, wrapping the single features
    out = tf.train.Example(features=tf.train.Features(feature=data))

    return out

  def write_images_to_tfr_short(self, input, output, filename:str="images"):
    filename= filename+".tfrecords"
    # Create a writer that'll store our data to disk
    writer = tf.io.TFRecordWriter(filename)
    count = 0

    for index in range(len(input)):

      #get the data we want to write
      current_input = input[index].astype('float32')
      current_output = output[index].astype('float32')

      out = self.parse_single_image(input_parse=current_input, output_parse=current_output)
      writer.write(out.SerializeToString())
      count += 1

    writer.close()
    print(f"Wrote {count} elements to TFRecord")
    return count

  
  def unison_shuffled_copies(self, a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]
  
  def apply_PCA(self, filename_flat, max_num_PC):

    file = tables.open_file(filename_flat, mode='w')
    atom = tables.Float32Atom()

    file.create_earray(file.root, 'data_flat', atom, (0, max_num_PC, 2))
    file.close()

    client = dask.distributed.Client(processes=False)

    N = int(self.n_samples * (self.num_sims))

    chunk_size = int(N/self.n_chunks)
    print('Applying incremental PCA ' + str(N//chunk_size) + ' times', flush = True)

    if (not os.path.isfile(filename_flat)) or (not os.path.isfile('ipca_input.pkl')):
      self.ipca_p = dask_ml.decomposition.IncrementalPCA(max_num_PC)
      self.ipca_input = dask_ml.decomposition.IncrementalPCA(max_num_PC)

      for i in range(int(N//chunk_size)):

        f = tables.open_file(self.filename, mode='r')
        x_array = f.root.data[i*chunk_size:(i+1)*chunk_size,:,:,0:3]
        obst_array = f.root.data[i*chunk_size:(i+1)*chunk_size,:,:,3:4]
        y_array = f.root.data[i*chunk_size:(i+1)*chunk_size,:,:,4:5]
        f.close()

        if x_array.shape[0] < max_num_PC:
          print('This chunck is too small ... skipping')
          break

        x_array_flat = x_array.reshape((x_array.shape[0], x_array.shape[1]*x_array.shape[2], x_array.shape[3]))
        # Normalize to [-1,1]
        x_array_flat1 = x_array_flat[...,0:1]/self.max_abs_Poisson_term_1
        x_array_flat2 = x_array_flat[...,1:2]/self.max_abs_delta_ux
        x_array_flat3 = x_array_flat[...,2:3]/self.max_abs_delta_uy

        obst_array_flat = obst_array.reshape((obst_array.shape[0], obst_array.shape[1]*obst_array.shape[2], 1 ))/self.max_abs_dist

        y_array_flat = y_array.reshape((y_array.shape[0], y_array.shape[1]*y_array.shape[2]))/self.max_abs_delta_p

        input_flat = np.concatenate((x_array_flat1, x_array_flat2, x_array_flat3, obst_array_flat) , axis = -1)
        input_flat = input_flat.reshape((input_flat.shape[0],-1))
        y_flat = y_array_flat.reshape((y_array_flat.shape[0],-1)) 

        # Scatter input and output data
        input_dask_future = client.scatter(input_flat)
        y_dask_future = client.scatter(y_array_flat)

        # Convert futures to Dask arrays
        input_dask = dask.array.from_delayed(input_dask_future, shape=input_flat.shape, dtype=input_flat.dtype)
        y_dask = dask.array.from_delayed(y_dask_future, shape=y_array_flat.shape, dtype=y_array_flat.dtype)

        #scaler = dask_ml.preprocessing.StandardScaler().fit(input_dask)
        scaler = dask_ml.preprocessing.StandardScaler().fit(input_dask.rechunk({1: input_dask.shape[1]}))

        #scaler1 = dask_ml.preprocessing.StandardScaler().fit(y_dask)
        scaler1 = dask_ml.preprocessing.StandardScaler().fit(y_dask.rechunk({1: y_dask.shape[1]}))

        inputScaled = scaler.transform(input_dask)
        yScaled = scaler1.transform(y_dask)

        self.ipca_input.partial_fit(inputScaled)
        self.ipca_p.partial_fit(yScaled)

        print('Fitted ' + str(i+1) + '/' + str(N//chunk_size), flush = True)

    else:
      print('Loading PCA arrays, as those are already available', flush=True)
      self.ipca_p = pk.load(open("ipca_p.pkl",'rb'))
      self.ipca_input = pk.load(open("ipca_input.pkl",'rb'))

    self.PC_p = np.argmax(self.ipca_p.explained_variance_ratio_.cumsum() > self.var_p) if np.argmax(self.ipca_p.explained_variance_ratio_.cumsum() > self.var_p) > 1 and np.argmax(self.ipca_p.explained_variance_ratio_.cumsum() > self.var_p) <= max_num_PC else max_num_PC  #max defined to be 32 here
    self.PC_input = np.argmax(self.ipca_input.explained_variance_ratio_.cumsum() > self.var_in) if np.argmax(self.ipca_input.explained_variance_ratio_.cumsum() > self.var_in) > 1 and np.argmax(self.ipca_input.explained_variance_ratio_.cumsum() > self.var_in) <= max_num_PC else max_num_PC

    print('PC_p :' + str(self.PC_p), flush = True)
    print('PC_input :' + str(self.PC_input), flush = True)

    print(' Total variance from input represented: ' + str(np.sum(self.ipca_input.explained_variance_ratio_[:self.PC_input])))
    pk.dump(self.ipca_input, open("ipca_input.pkl","wb"))

    print(' Total variance from p represented: ' + str(np.sum(self.ipca_p.explained_variance_ratio_[:self.PC_p])))
    pk.dump(self.ipca_p, open("ipca_p.pkl","wb"))

    for i in range(int(N//chunk_size)):

      with tables.open_file(self.filename, mode='r') as f:            
        x_array = f.root.data[i*chunk_size:(i+1)*chunk_size,:,:,0:3]
        obst_array = f.root.data[i*chunk_size:(i+1)*chunk_size,:,:,3:4]
        y_array = f.root.data[i*chunk_size:(i+1)*chunk_size,:,:,4:5]
        f.close()

      if x_array.shape[0] < max_num_PC:
        print('This chunck is too small ... skipping')
        break

      x_array_flat = x_array.reshape((x_array.shape[0], x_array.shape[1]*x_array.shape[2], x_array.shape[3]))
      # Normalize to [-1,1]
      x_array_flat1 = x_array_flat[...,0:1]/self.max_abs_Poisson_term_1
      x_array_flat2 = x_array_flat[...,1:2]/self.max_abs_delta_ux
      x_array_flat3 = x_array_flat[...,2:3]/self.max_abs_delta_uy

      obst_array_flat = obst_array.reshape((obst_array.shape[0], obst_array.shape[1]*obst_array.shape[2], 1 ))/self.max_abs_dist

      y_array_flat = y_array.reshape((y_array.shape[0], y_array.shape[1]*y_array.shape[2]))/self.max_abs_delta_p

      input_flat = np.concatenate((x_array_flat1, x_array_flat2, x_array_flat3, obst_array_flat) , axis = -1)
      input_flat = input_flat.reshape((input_flat.shape[0],-1))

      y_array_flat = y_array.reshape((y_array.shape[0], y_array.shape[1]*y_array.shape[2]))/self.max_abs_delta_p

      # Scatter input and output data
      input_dask_future = client.scatter(input_flat)
      y_dask_future = client.scatter(y_array_flat)
   
      # Convert futures to Dask arrays
      input_dask = dask.array.from_delayed(input_dask_future, shape=input_flat.shape, dtype=input_flat.dtype)
      y_dask = dask.array.from_delayed(y_dask_future, shape=y_array_flat.shape, dtype=y_array_flat.dtype)

      scaler = dask_ml.preprocessing.StandardScaler().fit(input_dask.rechunk({1: input_dask.shape[1]}))
      scaler1 = dask_ml.preprocessing.StandardScaler().fit(y_dask.rechunk({1: y_dask.shape[1]}))

      inputScaled = scaler.transform(input_dask)
      yScaled = scaler1.transform(y_dask)
      del scaler, scaler1
      
      input_transf = self.ipca_input.transform(input_flat)#[:,:self.PC_input]
      y_transf = self.ipca_p.transform(y_array_flat)#[:,:self.PC_input]

      array_image = np.concatenate((np.expand_dims(input_transf, axis=-1) , np.expand_dims(y_transf, axis=-1)), axis = -1)#, y_array]
      print(array_image.shape, flush = True)

      with tables.open_file(filename_flat, mode='a') as f:
        f.root.data_flat.append(np.array(array_image))
      
      del array_image
      del input_dask, y_dask, input_dask_future, y_dask_future
      del input_transf, y_transf
      del inputScaled, yScaled
      print('transformed ' + str(i+1) + '/' + str(N//chunk_size), flush = True)

    client.close()
    
  def prepare_data (self, hdf5_paths, max_num_PC, outarray_fn = 'outarray.h5', outarray_flat_fn= 'outarray_flat.h5'):

    #load data

    print_shape = True

    self.filename = outarray_fn
    filename_flat = outarray_flat_fn
      
    if not (os.path.isfile(outarray_fn) and os.path.isfile('maxs')):
      self.read_dataset()
    else:
      maxs = np.loadtxt('maxs')
      self.max_abs_Poisson_term_1, self.max_abs_delta_ux, self.max_abs_delta_uy, \
      self.max_abs_dist, \
      self.max_abs_delta_p = maxs
      
    if not (os.path.isfile(filename_flat) and os.path.isfile('ipca_input.pkl') and os.path.isfile('ipca_p.pkl')):
      print('Applying PCA \n')
      self.apply_PCA(filename_flat, max_num_PC)
    else:
      print('Data after PCA is available, load it and stepping over the PC analysis \n')
      self.ipca_p = pk.load(open("ipca_p.pkl",'rb'))
      self.ipca_input = pk.load(open("ipca_input.pkl",'rb'))

      self.PC_p = np.argmax(self.ipca_p.explained_variance_ratio_.cumsum() > self.var_p) if np.argmax(self.ipca_p.explained_variance_ratio_.cumsum() > self.var_p) > 1 and np.argmax(self.ipca_p.explained_variance_ratio_.cumsum() > self.var_p) <= max_num_PC else max_num_PC  #max defined to be 32 here
      self.PC_input = int(np.argmax(self.ipca_input.explained_variance_ratio_.cumsum() > self.var_in))

    f = tables.open_file(filename_flat, mode='r')
    input = f.root.data_flat[...,:self.PC_input,0] 
    output = f.root.data_flat[...,:self.PC_p,1] 
    f.close()

    # Treating PCA data

    if self.standardization_method == 'min_max':
      ## Option 2: Min-max scaling
      min_in = np.min(input, axis=0)
      max_in = np.max(input, axis=0)

      min_out = np.min(output, axis=0)
      max_out = np.max(output, axis=0)

      np.savez('min_max_values.npz', min_in=min_in, max_in=max_in, min_out=min_out, max_out=max_out)

      # Perform min-max scaling
      x = (input - min_in) / (max_in - min_in)
      y = (output - min_out) / (max_out - min_out)
      
    elif self.standardization_method == 'std':
      ## Option 1: Standardization
      mean_in = np.mean(input, axis=0)
      std_in = np.std(input, axis=0)

      mean_out = np.mean(output, axis=0)
      std_out = np.std(output, axis=0)

      np.savez('mean_std.npz', mean_in=mean_in, std_in=std_in, mean_out=mean_out, std_out=std_out)

      x = (input - mean_in) /std_in
      y = (output - mean_out) /std_out
    elif self.standardization_method == 'max_abs':

      # Option 3 - Old method
      max_abs_input_PCA = np.max(np.abs(input))
      max_abs_p_PCA = np.max(np.abs(output))
      print( max_abs_input_PCA, max_abs_p_PCA)

      np.savetxt('maxs_PCA', [max_abs_input_PCA, max_abs_p_PCA] )

      x = input/max_abs_input_PCA
      y = output/max_abs_p_PCA

    x, y = self.unison_shuffled_copies(x, y)
    print('Data shuffled \n')

    x = x.reshape((x.shape[0], x.shape[1], 1, 1))
    y = y.reshape((y.shape[0], y.shape[1], 1, 1))

    #tf records

    # Convert values to compatible tf.Example types.

    split = 0.9
    if not (os.path.isfile('train_data.tfrecords') and os.path.isfile('test_data.tfrecords')):
      count = self.write_images_to_tfr_short(x[:int(split*x.shape[0]),...], y[:int(split*y.shape[0]),...], filename="train_data")
      count = self.write_images_to_tfr_short(x[int(split*x.shape[0]):,...], y[int(split*y.shape[0]):,...], filename="test_data")
    else:
      print("TFRecords train and test data already available, using it... If you want to write new data, delete 'train_data.tfrecords' and 'test_data.tfrecords'!")
    self.len_train = int(split*x.shape[0])

    return 0 

  def parse_tfr_element(self, element):
    #use the same structure as above; it's kinda an outline of the structure we now want to create
    data = {
        'height': tf.io.FixedLenFeature([], tf.int64),
        'output' : tf.io.FixedLenFeature([], tf.string),
        'raw_input' : tf.io.FixedLenFeature([], tf.string),
        'depth_x':tf.io.FixedLenFeature([], tf.int64),
        'depth_y':tf.io.FixedLenFeature([], tf.int64)
      }

    content = tf.io.parse_single_example(element, data)
    
    height = content['height']
    depth_x = content['depth_x']
    depth_y = content['depth_y']
    output = content['output']
    raw_input = content['raw_input']
       
    #get our 'feature'-- our image -- and reshape it appropriately
    
    input_out= tf.io.parse_tensor(raw_input, out_type=tf.float32)
    output_out = tf.io.parse_tensor(output, out_type=tf.float32)

    return ( input_out , output_out)

  def load_dataset(self, filename, batch_size, buffer_size):
    #create the dataset
    dataset = tf.data.TFRecordDataset(filename)

      #pass every single feature through our mapping function
    dataset = dataset.map(
        self.parse_tfr_element
    )

    dataset = dataset.shuffle(buffer_size=buffer_size )
    #epoch = tf.data.Dataset.range(epoch_num)
    dataset = dataset.batch(batch_size)

    return dataset  
    
  def Callback_EarlyStopping(self, LossList, min_delta=0.1, patience=20):
      #No early stopping for 2*patience epochs
      if len(LossList)//patience < 2 :
          return False
      #Mean loss for last patience epochs and second-last patience epochs
      mean_previous = np.mean(LossList[::-1][patience:2*patience]) #second-last
      mean_recent = np.mean(LossList[::-1][:patience]) #last
      #you can use relative or absolute change
      delta_abs = np.abs(mean_recent - mean_previous) #abs change
      delta_abs = np.abs(delta_abs / mean_previous)  # relative change
      if delta_abs < min_delta :
          print("*CB_ES* Loss didn't change much from last %d epochs"%(patience))
          print("*CB_ES* Percent change in loss value:", delta_abs*1e2)
          return True
      else:
          return False


  def load_data_And_train(self, lr, batch_size, max_num_PC, model_name, beta_1, num_epoch, n_layers, width, dropout_rate, regularization, model_architecture, new_model):

    train_path = 'train_data.tfrecords'
    test_path = 'test_data.tfrecords'

    self.train_dataset = self.load_dataset(filename = train_path, batch_size= batch_size, buffer_size=1024)
    self.test_dataset = self.load_dataset(filename = test_path, batch_size= batch_size, buffer_size=1024)

    # Training 

    self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=beta_1, beta_2=0.999, epsilon=1e-08)#, decay=0.45*lr, amsgrad=True)
    self.loss_object = self.my_mse_loss()


    if new_model:
      if (model_architecture=='MLP_small') or (model_architecture=='MLP_big') or (model_architecture=='MLP_small_unet') or (model_architecture=='MLP_huge') or (model_architecture=='MLP_huger'):
        self.model = self.densePCA(n_layers, width, dropout_rate, regularization)
      elif model_architecture == 'conv1D':
        self.model = self.conv1D_PCA(n_layers, width, dropout_rate, regularization)
      elif model_architecture == 'MLP_attention':
        self.model = self.densePCA_attention(n_layers, width, dropout_rate, regularization)
      else:
        raise ValueError('Invalid NN model type')
    else:
      print(f"Loading model:{'model_' + model_name + '.h5'}")
      self.model = tf.keras.models.load_model('model_' + model_name + '.h5')

    epochs_val_losses, epochs_train_losses = [], []

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=beta_1, beta_2=0.999)#, epsilon=1e-08, decay=0.45*lr, amsgrad=True)

    min_yet = 1e9

    for epoch in range(num_epoch):
      progbar = tf.keras.utils.Progbar(math.ceil(self.len_train/batch_size))
      print('Start of epoch %d' %(epoch,))
      losses_train = []
      losses_test = []

      for step, (inputs, labels) in enumerate(self.train_dataset):

        inputs = tf.cast(inputs[...,0,0], dtype='float32')
        labels = tf.cast(labels[...,0,0], dtype='float32')
        loss = self.train_step(inputs, labels)
        losses_train.append(loss)

      losses_val  = self.perform_validation()

      losses_train_mean = np.mean(losses_train)
      losses_val_mean = np.mean(losses_val)

      epochs_train_losses.append(losses_train_mean)
      epochs_val_losses.append(losses_val_mean)
      print('Epoch %s: Train loss: %.4f , Validation Loss: %.4f \n' % (epoch,float(losses_train_mean), float(losses_val_mean)))

      progbar.update(step+1)

      # It was found that if the min_delta is too small, or patience is too high it can cause overfitting
      stopEarly = self.Callback_EarlyStopping(epochs_val_losses, min_delta=0.1/100, patience=25)
      if stopEarly:
        print("Callback_EarlyStopping signal received at epoch= %d/%d"%(epoch,num_epoch))
        break

      if epoch > 20:
        mod = 'model_' + model_name + '.h5'
        if losses_val_mean < min_yet:
          print(f'saving model: {mod}', flush=True)
          self.model.save(mod)
          min_yet = losses_val_mean
    
    print("Terminating training")
    mod = 'model_' + model_name + '.h5'
    ## Plot loss vs epoch
    plt.plot(list(range(len(epochs_train_losses))), epochs_train_losses, label ='train')
    plt.plot(list(range(len(epochs_val_losses))), epochs_val_losses, label ='val')
    plt.yscale('log')
    plt.legend()
    plt.savefig(f'loss_vs_epoch_beta{beta_1}lr{lr}reg{regularization}drop{dropout_rate}.png')

    ## Save losses data
    np.savetxt(f'train_loss_beta{beta_1}lr{lr}reg{regularization}drop{dropout_rate}.txt', epochs_train_losses, fmt='%d')
    np.savetxt(f'test_loss_beta{beta_1}lr{lr}reg{regularization}drop{dropout_rate}.txt', epochs_val_losses, fmt='%d')

        
    return 0
  
def main_train(dataset_path, num_sims, first_t, last_t, num_epoch, lr, beta, batch_size, \
              standardization_method, n_samples, block_size, delta, max_num_PC, \
              var_p, var_in, model_architecture, dropout_rate, outarray_fn, outarray_flat_fn, regularization, new_model, n_chunks, k):

  new_model = new_model.lower() == 'true'

  if model_architecture == 'MLP_small':
    n_layers = 3
    width = [512]*3
  elif model_architecture == 'MLP_big':
    n_layers = 7
    width = [256] + [512]*5 + [256]
  elif model_architecture == 'MLP_huge':
    n_layers = 12
    width = [256] + [512]*10 + [256]
  elif model_architecture == 'MLP_huger':
    n_layers = 20
    width = [256] + [512]*18 + [256]
  elif model_architecture == 'MLP_small_unet':
    n_layers = 9
    width = [512, 256, 128, 64, 32, 64, 128, 256, 512]
  elif model_architecture == 'conv1D':
    n_layers = 7
    width = [128, 64, 32, 16, 32, 64, 128]
  elif model_architecture == 'MLP_attention':
    n_layers = 3
    width = [512]*3
  else:
    raise ValueError('Invalid NN model type')

  paths = [dataset_path]
  num_sims = [num_sims]

  model_name = f'{model_architecture}-{standardization_method}-{var_p}-drop{dropout_rate}-lr{lr}-reg{regularization}-batch{batch_size}'

  Train = Training(delta, block_size,var_p, var_in, paths, n_samples, num_sims, first_t, last_t, standardization_method, n_chunks, k)

  # If you want to read the crude dataset (hdf5) again, delete the 'outarray.h5' file
  Train.prepare_data (paths, max_num_PC, outarray_fn, outarray_flat_fn) #prepare and save data to tf records
  Train.load_data_And_train(lr, batch_size, max_num_PC, model_name, beta, num_epoch, n_layers, width, dropout_rate, regularization, model_architecture, new_model)

if __name__ == '__main__':

  path_placa = 'dataset_plate_deltas_5sim20t.hdf5'
  dataset_path = [path_placa]

  num_sims_placa = 5
  first_t = 0
  last_t = num_sims_placa#, num_sims_rect, num_sims_tria, num_sims_placa]

  # Training Parameters
  num_epoch = 5000
  lr = 1e-5
  beta = 0.99
  batch_size = 1024 #*8
  ## Possible methods:
  ## 'std', 'min_max' or 'max_abs'
  standardization_method = 'std'

  # Data-Processing Parameters
  n_samples = int(1e4) #no. of samples per simulation
  block_size = 128
  delta = 5e-3
  max_num_PC = 512 # to not exceed the width of the NN
  var_p = 0.95
  var_in = 0.95

  model_architecture = 'MLP_small'
  dropout_rate = 0.1
  regularization = 1e-4

  outarray_fn = '../blocks_dataset/outarray.h5'
  outarray_flat_fn = '../blocks_dataset/outarray_flat.h5'

  new_model = True

  main_train(dataset_path, num_sims, first_t, last_t, num_epoch, lr, beta, batch_size, standardization_method, \
    n_samples, block_size, delta, max_num_PC, var_p, var_in, model_architecture, dropout_rate, outarray_fn, outarray_flat_fn, regularization, new_model)
