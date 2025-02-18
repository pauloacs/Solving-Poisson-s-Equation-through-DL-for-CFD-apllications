from numba import njit
import os
import shutil
import time
import h5py
import numpy as np

import matplotlib.pyplot as plt

import scipy.spatial.qhull as qhull


def interp_weights(xyz, uvw, d=2):
    """
    Get interpolation weights and vertices using barycentric interpolation.

    This function calculates the interpolation weights and vertices for interpolating values from the original grid to the target grid.
    The interpolation is performed using Delaunay triangulation.

    Args:
        xyz (ndarray): Coordinates of the original grid.
        uvw (ndarray): Coordinates of the target grid.
        d (int, optional): Number of dimensions. Default is 2.

    Returns:
        ndarray: Vertices of the simplices that contain the target grid points.
        ndarray: Interpolation weights for each target grid point.
    """
    tri = qhull.Delaunay(xyz)
    simplex = tri.find_simplex(uvw)
    vertices = np.take(tri.simplices, simplex, axis=0)
    temp = np.take(tri.transform, simplex, axis=0)
    delta = uvw - temp[:, d]
    bary = np.einsum('njk,nk->nj', temp[:, :d, :], delta)
    wts = np.hstack((bary, 1 - bary.sum(axis=1, keepdims=True)))
    valid = ~(simplex == -1)

    # Fill out-of-bounds points with Inverse-Distance Weighting
    if (~valid).any():
        tree = sklearn.neighbors.KDTree(xyz, leaf_size=40)
        nndist, nni = tree.query(np.array(uvw)[~valid], k=3)
        invalid = np.flatnonzero(~valid)
        vertices[invalid] = list(nni)
        wts[invalid] = list((1./np.maximum(nndist**2, 1e-6)) / (1./np.maximum(nndist**2, 1e-6)).sum(axis=-1)[:,None])
        
    return vertices, wts

def read_dataset(path, sim, time):
    """
    Reads dataset and splits it into the internal flow data (data) and boundary data.

    Args:
        path (str): Path to hdf5 dataset
        sim (int): Simulation number.
        time (int): Time frame.
    """
    with h5py.File(path, "r") as f:
        data = f["sim_data"][sim:sim+1, time:time+1, ...]
        top_boundary = f["top_bound"][sim:sim+1, time:time+1, ...]
        obst_boundary = f["obst_bound"][sim:sim+1, time:time+1, ...]

    return data, top_boundary, obst_boundary


#@njit
def interpolate_fill(values, vtx, wts, fill_value=np.nan):
    """
    Interpolate based on previously computed vertices (vtx) and weights (wts) and fill.

    Args:
        values (NDArray): Array of values to interpolate.
        vtx (NDArray): Array of interpolation vertices.
        wts (NDArray): Array of interpolation weights.
        fill_value (float): Value used to fill.
    
    Returns:
        NDArray: Interpolated values with fill_value for invalid weights.
    """
    ret = np.einsum('nj,nj->n', np.take(values, vtx), wts)
    ret[np.any(wts < 0, axis=1)] = fill_value
    return ret


#@njit(nopython = True)  #much faster using numba.njit but is giving an error
def index(array, item):
    """
    Finds the index of the first element equal to item.

    Args:
        array (NDArray):
        item (float):
    """
    for idx, val in np.ndenumerate(array):
        if val == item:
            return idx
        # else:
        # 	return None
    # If no item was found return None, other return types might be a problem due to
    # numbas type inference.


def create_uniform_grid(x_min, x_max, y_min, y_max, delta):
    """
    Creates an uniform 2D grid (should envolve every cell of the mesh).

    Args:
        x_min (float): The variable name is self-explanatory.
        x_max (float): The variable name is self-explanatory.
        y_min (float): The variable name is self-explanatory.
        y_max (float): The variable name is self-explanatory.
    """
    X0 = np.linspace(x_min + delta/2 , x_max - delta/2 , num = int(round( (x_max - x_min)/delta )) )
    Y0 = np.linspace(y_min + delta/2 , y_max - delta/2 , num = int(round( (y_max - y_min)/delta )) )

    XX0, YY0 = np.meshgrid(X0,Y0)
    return XX0.flatten(), YY0.flatten()


def createGIF(n_sims, n_ts):
    
    ####################### TO CREATE A GIF WITH ALL THE FRAMES ###############################
    filenamesp = []

    for sim in range(5):
        for time in range(5):
            filenamesp.append(f'plots/p_pred_sim{sim}t{time}.png') #hardcoded to get the frames in order

    import imageio

    with imageio.get_writer('plots/p_movie.gif', mode='I', duration =0.5) as writer:
        for filename in filenamesp:
            image = imageio.imread(filename)
            writer.append_data(image)
    ######################## ---------------- //----------------- ###################

def plot_random_blocks(res_concat, y_array, x_array, sim, time, save_plots):
    """
    Plot 9 randomly sampled blocks for reference.

    Args:
        res_concat (ndarray): The array containing the predicted flow fields for each block.
        y_array (ndarray): The array containing the ground truth flow fields for each block.
        x_array (ndarray): The array containing the input flow fields for each block.
        sim (int): The simulation number.
        time (int): The time step number.
        save_plots (bool): Whether to save the plots.

    Returns:
        None
    """
    if save_plots:
        # plot blocks
        N = res_concat.shape[0]  # Number of blocks

        # Select 9 random indices
        random_indices = np.random.choice(N, size=9, replace=False)

        # Create the figure and axes for a 3x6 grid (3x3 for each side)
        fig, axes = plt.subplots(3, 6, figsize=(18, 12))

        # Add big titles for the left and right 3x3 grids with larger font size
        fig.text(0.25, 0.92, "SM Predictions", ha="center", fontsize=18, fontweight='bold')
        fig.text(0.75, 0.92, "CFD Predictions (Ground Truth)", ha="center", fontsize=18, fontweight='bold')

        for idx, i in enumerate(random_indices):
            row = idx // 3
            col = idx % 3

            # Plot SM predictions (left 3x3 grid)
            ax_sm = axes[row, col]
            masked_arr = np.ma.array(res_concat[i,:,:,0], mask=x_array[i,:,:,2]!=0)
            ax_sm.imshow(masked_arr, cmap='viridis')
            ax_sm.set_title(f"Block {i}/{N}", fontsize=12, fontweight='bold')
            ax_sm.axis("off")

            # Add border around each block for clearer distinction
            for _, spine in ax_sm.spines.items():
                spine.set_edgecolor('black')
                spine.set_linewidth(2)

            # Plot CFD predictions (right 3x3 grid)
            ax_cfd = axes[row, col + 3]
            masked_arr = np.ma.array(y_array[i,:,:,0], mask=x_array[i,:,:,2]!=0)
            ax_cfd.imshow(masked_arr, cmap='viridis')
            ax_cfd.set_title(f"Block {i}/{N}", fontsize=12, fontweight='bold')
            ax_cfd.axis("off")

            # Add border around each block for clearer distinction
            for _, spine in ax_cfd.spines.items():
                spine.set_edgecolor('black')
                spine.set_linewidth(2)

        # Adjust layout to make space for titles
        plt.tight_layout(rect=[0, 0, 1, 0.88])

        # Save the plot as an image file
        output_path = f"plots/sim{sim}/SM_vs_CFD_predictions_t{time}.png"  # Change the filename/path as needed
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

def compute_in_block_error(pred, true, flow_bool):

    true_masked = true[flow_bool]
    pred_masked = pred[flow_bool]

    # Calculate norm based on reference data a predicted data
    norm_true = np.max(true_masked) - np.min(true_masked)
    norm_pred = np.max(pred_masked) - np.min(pred_masked)

    norm = norm_true

    mask_nan = ~np.isnan( pred_masked  - true_masked )

    BIAS_norm = np.mean( (pred_masked  - true_masked )[mask_nan] )/norm * 100
    RMSE_norm = np.sqrt(np.mean( ( pred_masked  - true_masked )[mask_nan]**2 ))/norm * 100
    STDE_norm = np.sqrt( (RMSE_norm**2 - BIAS_norm**2) )
    
    print(f"""
            norm_true = {norm_true};
            norm_pred = {norm_pred};

    ** Error in delta_p (blocks) **

        normVal  = {norm} Pa
        biasNorm = {BIAS_norm:.3f}%
        stdeNorm = {STDE_norm:.3f}%
        rmseNorm = {RMSE_norm:.3f}%
    """, flush = True)

    pred_minus_true_block = np.mean( (pred_masked  - true_masked )[mask_nan] )/norm
    pred_minus_true_squared_block = np.mean( (pred_masked  - true_masked )[mask_nan]**2 )/norm**2
    return pred_minus_true_block, pred_minus_true_squared_block