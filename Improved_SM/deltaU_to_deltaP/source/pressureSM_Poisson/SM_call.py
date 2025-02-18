import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU') 
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)

from numba import njit
import os
import shutil
import time
import h5py
import numpy as np

import pickle as pk
import matplotlib.pyplot as plt
import matplotlib.path as mpltPath

from shapely.geometry import MultiPoint
from scipy.spatial import distance
import scipy.spatial.qhull as qhull
import scipy.ndimage as ndimage

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

class Evaluation():
	def __init__(self, delta, shape, overlap, var_p, var_in, dataset_path, model_path, max_num_PC, standardization_method, k, phis_fn):
		"""
		Initialize Evaluation class. 

		Args:
			delta (float): The value of delta.
			shape (int): The shape value.
			overlap (float): The overlap value.
			var_p (float): The var_p value.
			var_in (float): The var_in value.
			dataset_path (str): The path to the dataset.
			model_path (str): The path to the model.
			max_num_PC (int): The maximum number of principal components.
			standardization_method (str): The standardization method.
			k (float):

		Attributes:
			delta (float): The value of delta.
			shape (int): The shape value.
			overlap (float): The overlap value.
			var_p (float): The var_p value.
			var_in (float): The var_in value.
			dataset_path (str): The path to the dataset.
			model_path (str): The path to the model.
			max_num_PC (int): The maximum number of principal components.
			standardization_method (str): The standardization method.
			k (float): Number of standard deviations to smooth the high variance fields.
			max_abs_delta_Ux (float): The maximum absolute value of delta Ux.
			max_abs_delta_Uy (float): The maximum absolute value of delta Uy.
			max_abs_dist (float): The maximum absolute value of dist.
			max_abs_delta_p (float): The maximum absolute value of p.
			model (tf.keras.Model): The loaded model.
			pcainput (pkl): The loaded pca input.
			pcap (pkl): The loaded pca p.
			pc_p (int): The number of principal components for p.
			pc_in (int): The number of principal components for input.
		"""
		self.delta = delta
		self.shape = shape
		self.overlap = overlap
		self.var_in = var_in
		self.var_p = var_p
		self.dataset_path = dataset_path
		self.standardization_method = standardization_method
		self.k = k
		self.phis_fn = phis_fn

		maxs = np.loadtxt('maxs')

		self.max_abs_Poisson_term_1, self.max_abs_delta_Ux, self.max_abs_delta_Uy, self.max_abs_dist, self.max_abs_delta_p = maxs

		#### loading the model #######
		if 'MLP_attention_biased' in model_path:
			from pressureSM_deltas.train import BiasedAttention
			self.model = tf.keras.models.load_model(model_path, custom_objects={'BiasedAttention': BiasedAttention})
		else:
			self.model = tf.keras.models.load_model(model_path)
		print(self.model.summary())
		
		### loading the pca matrices for transformations ###
		self.pcainput = pk.load(open("ipca_input.pkl",'rb'))
		self.pcap = pk.load(open("ipca_p.pkl",'rb'))

		self.pc_p = np.argmax(self.pcap.explained_variance_ratio_.cumsum() > self.var_p) if np.argmax(self.pcap.explained_variance_ratio_.cumsum() > self.var_p) > 1 and np.argmax(self.pcap.explained_variance_ratio_.cumsum() > self.var_p) <= max_num_PC else max_num_PC
		self.pc_in = np.argmax(self.pcainput.explained_variance_ratio_.cumsum() > self.var_in) if np.argmax(self.pcainput.explained_variance_ratio_.cumsum() > self.var_in) > 1 and np.argmax(self.pcainput.explained_variance_ratio_.cumsum() > self.var_in) <= max_num_PC else max_num_PC


	def interp_weights(self, xyz, uvw, d=2):
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

	#@njit
	def interpolate_fill(self, values, vtx, wts, fill_value=np.nan):
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
	
	def create_uniform_grid(self, x_min, x_max, y_min, y_max):
		"""
		Creates an uniform 2D grid (should envolve every cell of the mesh).

        Args:
            x_min (float): The variable name is self-explanatory.
            x_max (float): The variable name is self-explanatory.
            y_min (float): The variable name is self-explanatory.
			y_max (float): The variable name is self-explanatory.
		"""
		X0 = np.linspace(x_min + self.delta/2 , x_max - self.delta/2 , num = int(round( (x_max - x_min)/self.delta )) )
		Y0 = np.linspace(y_min + self.delta/2 , y_max - self.delta/2 , num = int(round( (y_max - y_min)/self.delta )) )

		XX0, YY0 = np.meshgrid(X0,Y0)
		return XX0.flatten(), YY0.flatten()

	#@njit(nopython = True)  #much faster using numba.njit but is giving an error
	def index(self, array, item):
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

	def read_dataset(self, path, sim, time):
		"""
		Reads dataset and splits it into the internal flow data (data) and boundary data.

		Args:
			path (str): Path to hdf5 dataset
			sim (int): Simulation number.
			time (int): Time frame.
		"""
		with h5py.File(path, "r") as f:
			data = f["sim_data"][sim:sim+1,time:time+1, ...]
			top_boundary = f["top_bound"][sim:sim+1, time:time+1 , ...]
			obst_boundary = f["obst_bound"][sim:sim+1, time:time+1 , ...]
		return data, top_boundary, obst_boundary

	def computeOnlyOnce(self, sim):
		"""
		Performs interpolation from the OF grid (corresponding to the mesh cell centers),
		saves the intepolation vertices and weights and computes the signed distance function (sdf).

		Args:
			sim (int): Simulation number.
		"""
		time = 0
		data, top_boundary, obst_boundary = self.read_dataset(self.dataset_path, sim , time)

		self.indice = self.index(data[0,0,:,0] , -100.0 )[0]


		x_min = round(np.min(data[0,0,...,:self.indice,3]),3) 
		x_max = round(np.max(data[0,0,...,:self.indice,3]),3) 

		y_min = round(np.min(data[0,0,...,:self.indice,4]),3)  #- 0.3
		y_max = round(np.max(data[0,0,...,:self.indice,4]),3)  #+ 0.3

		######### -------------------- Assuming constant mesh, the following can be done out of the for cycle ------------------------------- ##########

		X0, Y0 = self.create_uniform_grid(x_min, x_max, y_min, y_max)
		self.X0 = X0
		self.Y0 = Y0
		xy0 = np.concatenate((np.expand_dims(X0, axis=1),np.expand_dims(Y0, axis=1)), axis=-1)
		points = data[0,0,:self.indice,3:5] #coordinates
		self.vert, self.weights = self.interp_weights(points, xy0) #takes ~100% of the interpolation cost and it's only done once for each different mesh/simulation case

		# boundaries indice
		indice_top = self.index(top_boundary[0,0,:,0] , -100.0 )[0]
		top = top_boundary[0,0,:indice_top,:]
		self.max_x, self.max_y = np.max([(top[:,0]).max(), x_max]), np.min([(top[:,1]).max(), y_max])
		self.min_x, self.min_y = np.max([(top[:,0]).min(), x_min]), np.min([(top[:,1]).min(), y_min])

		is_inside_domain = ( xy0[:,0] <= self.max_x)  * ( xy0[:,0] >= self.min_x ) * ( xy0[:,1] <= self.max_y ) * ( xy0[:,1] >= self.min_y ) #rhis is just for simplification

		indice_obst = self.index(obst_boundary[0,0,:,0] , -100.0 )[0]
		obst = obst_boundary[0,0,:indice_obst,:]

		obst_points = MultiPoint(obst)

		# This only works for convex geometries
		hull = obst_points.convex_hull  
		hull_pts = hull.exterior.coords.xy  
		hull_pts = np.c_[hull_pts[0], hull_pts[1]]

		path = mpltPath.Path(hull_pts)
		is_inside_obst = path.contains_points(xy0)

		domain_bool = is_inside_domain * ~is_inside_obst

		top = top[0:top.shape[0]:5,:]   #if this has too many values, using cdist can crash the memmory since it needs to evaluate the distance between ~1M points with thousands of points of top
		obst = obst[0:obst.shape[0]:5,:]

		sdf = np.minimum(distance.cdist(xy0,obst).min(axis=1), distance.cdist(xy0,top).min(axis=1) ) * domain_bool

		#div defines the sliding window vertical and horizontal displacements
		div = 1 

		self.grid_shape_y = int(round((y_max-y_min)/self.delta)) #+1
		self.grid_shape_x = int(round((x_max-x_min)/self.delta)) #+1

		i = 0
		j = 0

		#arrange data in array: #this can be put outside the j loop if the mesh is constant 

		x0 = np.min(X0)
		y0 = np.min(Y0)
		dx = self.delta
		dy = self.delta

		indices= np.zeros((X0.shape[0],2))
		obst_bool = np.zeros((self.grid_shape_y,self.grid_shape_x,1))
		self.sdfunct = np.zeros((self.grid_shape_y,self.grid_shape_x,1))

		p = data[i,j,:self.indice,2:3] #values
		p_interp = self.interpolate_fill(p, self.vert, self.weights) 

		for (step, x_y) in enumerate(xy0):  
			if domain_bool[step] * (~np.isnan(p_interp[step])) :
				jj = int(round((x_y[...,0] - x0) / dx))
				ii = int(round((x_y[...,1] - y0) / dy))

				indices[step,0] = ii
				indices[step,1] = jj
				self.sdfunct[ii,jj,:] = sdf[step]
				obst_bool[ii,jj,:]  = int(1)

		self.indices = indices.astype(int)

		return 0

	def assemble_prediction(self, array, indices_list, n_x, n_y, apply_filter, shape_x, shape_y, deltaU_change_grid, deltaP_prev_grid, apply_deltaU_change_wgt):
		"""
		Reconstructs the flow domain based on squared blocks.
		In the first row the correction is based on the outlet fixed value BC.
		
		In the following rows the correction is based on the overlap region at the top of each new block.
		This correction from the top ensures better agreement between different rows, leading to overall better results.

		Args:
			array (ndarray): The array containing the predicted flow fields for each block.
			indices_list (list): The list of indices representing the position of each block in the flow domain.
			n_x (int): The number of blocks in the x-direction.
			n_y (int): The number of blocks in the y-direction.
			apply_filter (bool): Flag indicating whether to apply a Gaussian filter to remove boundary artifacts.
			shape_x (int): The width of each block.
			shape_y (int): The height of each block.

		Returns:
			ndarray: The reconstructed flow domain.

		"""
		overlap = self.overlap
		shape = self.shape
		Ref_BC = self.Ref_BC

		result_array = np.empty(shape=(shape_y, shape_x))

		## Array to store the average pressure in the overlap region with the next down block
		BC_ups = np.zeros(n_x+1)

		# i index where the lower blocks are located
		p_i = shape_y - ( (shape-overlap) * n_y + shape )
		
		# j index where the left-most blocks are located
		p_j = shape_x - ( (shape - overlap) * n_x + shape )

		result = result_array

		## Loop over all the blocks and apply corrections to ensure consistency between overlapping blocks
		for i in range(self.x_array.shape[0]):

			idx_i, idx_j = indices_list[i]
			flow_bool = self.x_array[i,:,:,2]
			pred_field = array[i,...]

			## FIRST row
			if idx_i == 0:

				## Calculating correction to be applied
				if i == 0: 
 					## First correction - based on the outlet fixed pressure boundary condition
					BC_coor = np.mean(pred_field[:,-1][flow_bool[:,-1]!=0]) - Ref_BC  # i = 0 sits outside the inclusion zone
				else:
					BC_ant_0 = np.mean(old_pred_field[:,:overlap][flow_bool[:,:overlap] !=0]) 
					BC_coor = np.mean(pred_field[:,-overlap:][flow_bool[:,-overlap:]!=0]) - BC_ant_0
				if idx_j == 0:
					intersect_zone_limit = overlap - p_j
					BC_ant_0 = np.mean(old_pred_field[:, :intersect_zone_limit][flow_bool[:, :intersect_zone_limit] !=0]) 
					BC_coor = np.mean(pred_field[:, -intersect_zone_limit:][flow_bool[:, -intersect_zone_limit:]!=0]) - BC_ant_0

				## Applying correction
				pred_field -= BC_coor

				## Storing upward average pressure
				BC_ups[idx_j] = np.mean(pred_field[-overlap:,:][flow_bool[-overlap:,:] !=0])

			## MIDDLE rows
			elif idx_i != n_y + 1:

				## Calculating correction to be applied
				if np.isnan(BC_ups[idx_j]): #### THIS PART IS NOT WORKING WELL ... CORRECT THIS!!
					if idx_j == 0:
						intersect_zone_limit = overlap - p_j
						BC_ant_0 = np.mean(old_pred_field[:, :intersect_zone_limit][flow_bool[:, :intersect_zone_limit] !=0]) 
						BC_coor = np.mean(pred_field[:, -intersect_zone_limit:][flow_bool[:, -intersect_zone_limit:]!=0]) - BC_ant_0
					elif idx_j == n_x:
						# Here it always needs to be corrected from above to keep consistency
						BC_ant_0 = np.mean(old_pred_field[:,:overlap][flow_bool[:,:overlap] !=0]) 
						BC_coor = np.mean(pred_field[:overlap,:][flow_bool[:overlap,:]!=0]) - BC_ups[idx_j]
					else:
						BC_ant_0 = np.mean(old_pred_field[:,:overlap][flow_bool[:,:overlap] !=0]) 
						BC_coor = np.mean(pred_field[:,-overlap:][flow_bool[:,-overlap:]!=0]) - BC_ant_0											
				else:
					BC_coor = np.mean(pred_field[:overlap,:][flow_bool[:overlap,:]!=0]) - BC_ups[idx_j]
				
					# if idx_j != 0 and idx_j != n_x:
					# 	BC_ant_0 = np.mean(old_pred_field[:,:overlap][flow_bool[:,:overlap] !=0]) 
					# 	BC_coor_2 = np.mean(pred_field[:,-overlap:][flow_bool[:,-overlap:]!=0]) - BC_ant_0
					
					# 	## Apply the lowest correction ... less prone to problems...
					# 	if abs(BC_coor_2) *5 < abs(BC_coor):
					# 		BC_coor = BC_coor_2

				## Applying correction
				pred_field -= BC_coor

				## Storing upward average pressure
				BC_ups[idx_j] = np.mean(pred_field[-overlap:,:][flow_bool[-overlap:,:] !=0])
				
				## Value stored to be used in the last row depends on p_i
				if idx_i == n_y:
					BC_ups[idx_j] = np.mean(pred_field[-(shape-p_i):,:][flow_bool[-(shape-p_i):,:] !=0])
			
			## LAST row
			else:

				## Calculating correction to be applied

				## In the last column the correction needs to be from above to keep consistency (with BC_ups)
				if idx_j == n_x:
					BC_coor = np.mean(pred_field[-p_i-overlap:-p_i,:][flow_bool[-p_i-overlap:-p_i,:]!=0]) - BC_ups[idx_j]
				else:
									
					#up
					y_0 = -p_i - overlap
					y_f = -p_i
					n_up_non_nans = (flow_bool[y_0:y_f,:]!=0).sum()
					# right side
					x_0 = shape_x -shape - (n_x-1)*(shape-overlap)
					n_right_non_nans = (flow_bool[:, x_0:]!=0).sum()

					# Give preference to "up" or "right" correction???
					## Giving it to "up" because it is being done everywhere else
					## only switch method if in the overlap region more than 90% of the values are NANs
					
					if (n_up_non_nans)/128**2 > 0.9:
						if idx_j == 0:
							intersect_zone_limit = overlap - p_j
							BC_ant_0 = np.mean(old_pred_field[:, :intersect_zone_limit][flow_bool[:, :intersect_zone_limit] !=0]) 
							BC_coor = np.mean(pred_field[:, -intersect_zone_limit:][flow_bool[:, -intersect_zone_limit:]!=0]) - BC_ant_0
						else:
							BC_ant_0 = np.mean(old_pred_field[:,:overlap][flow_bool[:,:overlap] !=0]) 
							BC_coor = np.mean(pred_field[:,-overlap:][flow_bool[:,-overlap:]!=0]) - BC_ant_0								
					else:
						BC_coor = np.mean(pred_field[:-p_i,:][flow_bool[:-p_i,:]!=0]) - BC_ups[idx_j]

						# if idx_j != 0:
						# 	BC_ant_0 = np.mean(old_pred_field[:,:overlap][flow_bool[:,:overlap] !=0]) 
						# 	BC_coor_2 = np.mean(pred_field[:,-overlap:][flow_bool[:,-overlap:]!=0]) - BC_ant_0
						
						# 	## Apply the lowest correction ... less prone to problems...
						# 	if abs(BC_coor_2) *5 < abs(BC_coor):
						# 		BC_coor = BC_coor_2
								

				## Applying the correction
				pred_field -= BC_coor
				
			old_pred_field = pred_field
			
			## Last reassembly step:
			## Assigning the block to the right location in the flow domain
			if [idx_i, idx_j] == [n_y + 1, 0]:
				result[-p_i:shape_y , 0:shape] = pred_field[-p_i:]
			elif idx_j == 0:
				result[(shape-overlap) * idx_i:(shape-overlap) * idx_i + shape, 0:shape] = pred_field
			## Last row
			elif idx_i == (n_y + 1):
				idx_j = n_x - idx_j
				# option 1 - thick penultimate row
				result[-p_i:, shape_x -shape - idx_j*(shape-overlap) :  shape_x- idx_j*(shape-overlap)] = pred_field[-p_i:]
				# option 2 - thick last row
				#result[-shape:, shape_x -shape - idx_j*(shape-overlap) :  shape_x- idx_j*(shape-overlap)] = pred_field

			else:
				idx_j = n_x - idx_j
				result[(shape-overlap) * idx_i:(shape-overlap) * idx_i + shape, shape_x -shape - idx_j*(shape-overlap) : shape_x- idx_j*(shape-overlap)] = pred_field
				
		result -= np.mean( 3* result[:,-1] - result[:,-2] )/3

		################### this applies a gaussian filter to remove boundary artifacts #################
		filter_tuple = (10, 10)

		if apply_filter:
			result = ndimage.gaussian_filter(result, sigma=filter_tuple, order=0)

		change_in_deltap = None
		if apply_deltaU_change_wgt:
			deltaU_change_grid = ndimage.gaussian_filter(deltaU_change_grid, sigma=(50,50), order=0)
			change_in_deltap = result - deltaP_prev_grid
			change_in_deltap = change_in_deltap * deltaU_change_grid
			change_in_deltap = ndimage.gaussian_filter(change_in_deltap, sigma=filter_tuple, order=0)

		return result, change_in_deltap

	def timeStep(self, sim, time, plot_intermediate_fields, save_plots, show_plots, apply_filter, phi):
		"""
		Performs a time step in the simulation.

		Args:
			sim (int): The simulation number.
			time (int): The time step number.
			plot_intermediate_fields (bool): Whether to plot intermediate fields during the time step.
			save_plots (bool): Whether to save the plots.
			show_plots (bool): Whether to display the plots.
			apply_filter: The filter to apply.
			phi (float): Simulation characteristic lenght.

		Returns:
			None
		"""
		data, top_boundary, obst_boundary = self.read_dataset(self.dataset_path, sim , time)
		i = 0
		j = 0
		Ux =  data[i,j,:self.indice,0:1] #values
		Uy =  data[i,j,:self.indice,1:2] #values

		delta_U = data[i,j,:self.indice,5:7] #values
		delta_Ux = delta_U[...,0:1]
		delta_Uy = delta_U[...,1:2]

		delta_U_prev = data[i,j,:self.indice, 8:10] #values
		delta_p_prev = data[i,j,:self.indice,10:11] #values

		# check where the deltaU has changed in the last time step
		deltaU_changed = np.abs(delta_U - delta_U_prev).sum(axis=-1)
		deltaU_changed = deltaU_changed / deltaU_changed.max()

		# For accuracy accessment
		delta_p = data[i,j,:self.indice,7:8] #values
		p = data[i,j,:self.indice,2:3] #values

		U_max_norm = np.max(np.sqrt(np.square(Ux) + np.square(Uy)))
		deltaU_max_norm = np.max(np.sqrt(np.square(delta_Ux) + np.square(delta_Uy)))

		# Ignore time steps with minimal changes ...
		# there is not point in computing error metrics for these
		# it would exagerate the delta_p errors and give ~0% errors in p
		threshold = 1e-4
		irrelevant_ts = (deltaU_max_norm/U_max_norm) < threshold or deltaU_max_norm < 1e-6 or U_max_norm < 1e-6

		if irrelevant_ts:
			print(f"\n\n Irrelevant time step, U_norm changed {(deltaU_max_norm/U_max_norm)*100:.2f} [%] (criteria: < {threshold*100:.2f} [%]).\n Skipping Time step.")
			return 0

		# Representative scales
		L = phi
		#U = 1
		U = U_max_norm
		#Re = phi * U/ nu

		# The derivation ca be done from the the data points - very heavy
		# or from the python grid - cheap

		#p_prev_interp = self.interpolate_fill(p_prev, vert, weights)
		Ux_interp = self.interpolate_fill(Ux, self.vert, self.weights)
		Uy_interp = self.interpolate_fill(Uy, self.vert, self.weights)
		delta_Ux_interp = self.interpolate_fill(delta_Ux, self.vert, self.weights)
		delta_Uy_interp = self.interpolate_fill(delta_Uy, self.vert, self.weights)
		delta_p_interp = self.interpolate_fill(delta_p, self.vert, self.weights)
		p_interp = self.interpolate_fill(p, self.vert, self.weights)
		# weighting 
		deltaU_changed_interp = self.interpolate_fill(deltaU_changed, self.vert, self.weights)
		delta_p_prev_interp = self.interpolate_fill(delta_p_prev, self.vert, self.weights)

		# 1D vectors to 2D arrays
		ux_grid = np.zeros(shape=(self.grid_shape_y, self.grid_shape_x))
		uy_grid = np.zeros(shape=(self.grid_shape_y, self.grid_shape_x))
		#p_prev_grid = np.zeros(shape=(self.grid_shape_y, self.grid_shape_x))
		delta_ux_grid = np.zeros(shape=(self.grid_shape_y, self.grid_shape_x))
		delta_uy_grid = np.zeros(shape=(self.grid_shape_y, self.grid_shape_x))
		#
		ux_grid[tuple(self.indices.T)] = Ux_interp.reshape(Ux_interp.shape[0])
		uy_grid[tuple(self.indices.T)] = Uy_interp.reshape(Uy_interp.shape[0])
		#p_prev_grid[tuple(self.indices.T)] = p_prev_interp.reshape(p_prev_interp.shape[0])
		delta_ux_grid[tuple(self.indices.T)] = delta_Ux_interp.reshape(delta_Ux_interp.shape[0])
		delta_uy_grid[tuple(self.indices.T)] = delta_Uy_interp.reshape(delta_Uy_interp.shape[0])

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

		#p_prev_grid[sdfunct[...,0]==0] = np.nan
		ux_grid[self.sdfunct[...,0]==0] = np.nan
		uy_grid[self.sdfunct[...,0]==0] = np.nan

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

		delta_Ux = delta_ux_grid / U
		delta_Uy = delta_uy_grid / U
		#p_prev = p_prev_grid / U**2

		Poisson_term_1_f = smart_arcsin_smooth_transform(Poisson_term_1, self.k)

		##################################################
		### Plot to show filtering...
		##################################################
		import matplotlib.colors as mcolors
		# Define normalization to center at 0
		#norm = mcolors.TwoSlopeNorm(vmin=Poisson_term_1.min(), vcenter=0, vmax=Poisson_term_1.max())
		#norm_filtered = mcolors.TwoSlopeNorm(vmin=Poisson_term_1_f.min(), vcenter=0, vmax=Poisson_term_1_f.max())
		fig, axes = plt.subplots(2, 2, figsize=(12, 3))

		# Plot the original term
		im1 = axes[0, 0].imshow(smart_arcsin_smooth_transform(Poisson_term_1, 10), cmap='RdBu', aspect='auto')
		axes[0, 0].set_title(r'$-\nabla^* \cdot ((u^* \cdot \nabla^*) u^*)$')
		axes[0, 0].set_xticks([])
		axes[0, 0].set_yticks([])

		# Plot the transformed term
		im2 = axes[0, 1].imshow(Poisson_term_1_f, cmap='RdBu', aspect='auto')
		axes[0, 1].set_title(r'$-\nabla^* \cdot ((u^* \cdot \nabla^*) u^*)$ after arcsinh transformation')
		axes[0, 1].set_xticks([])
		axes[0, 1].set_yticks([])

		# Plot histograms with automatic limits
		axes[1, 0].hist(Poisson_term_1.flatten(), bins=250, color='grey', log=True)
		axes[1, 0].set_xlim(Poisson_term_1.min(), Poisson_term_1.max())  # Set limits to actual data range
		axes[1, 0].set_title('Values distribution')
		axes[1, 0].set_xticks([])
		axes[1, 0].set_yticks([])

		axes[1, 1].hist(Poisson_term_1_f.flatten(), bins=250, color='grey', log=True)
		axes[1, 1].set_xlim(Poisson_term_1_f.min(), Poisson_term_1_f.max())  # Set limits to actual data range
		axes[1, 1].set_title('Values distribution after arcsinh transformation')
		axes[1, 1].set_xticks([])
		axes[1, 1].set_yticks([])

		plt.tight_layout()
		plt.savefig('arcsin_transf.png')
		plt.show()
		plt.close()
		################################################################################

		# The output
		delta_p_adim = delta_p_interp/pow(U,2.0) 
		delta_p_grid = np.zeros(shape=(self.grid_shape_y, self.grid_shape_x))
		delta_p_grid[tuple(self.indices.T)] = delta_p_adim.reshape(delta_p_adim.shape[0])
		p_grid = np.zeros(shape=(self.grid_shape_y, self.grid_shape_x))
		p_grid[tuple(self.indices.T)] = p_interp.reshape(p_interp.shape[0])

		# # 1D vectors to 2D arrays
		grid = np.zeros(shape=(1, self.grid_shape_y, self.grid_shape_x, 6))
		grid[0,:,:,0] = Poisson_term_1_f
		grid[0,:,:,1] = delta_Ux
		grid[0,:,:,2] = delta_Uy
		grid[0,:,:,3] = self.sdfunct[...,0]
		grid[0,:,:,4] = delta_p_grid
		grid[0,:,:,5] = p_grid	

		grid[np.isnan(grid)] = 0 #set any nan value to 0

		## Rescale all the variables to [-1,1]
		grid[0,:,:,0] /= self.max_abs_Poisson_term_1
		grid[0,:,:,1] /= self.max_abs_delta_Ux
		grid[0,:,:,2] /= self.max_abs_delta_Uy
		grid[0,:,:,3] /= self.max_abs_dist
		grid[0,:,:,4] /= self.max_abs_delta_p

		# saving for weighting procedure
		deltaU_change_grid = np.zeros(shape=(self.grid_shape_y, self.grid_shape_x))
		deltaU_change_grid[tuple(self.indices.T)] = deltaU_changed_interp.reshape(deltaU_changed_interp.shape[0])
		deltaP_prev_grid = np.zeros(shape=(self.grid_shape_y, self.grid_shape_x))
		deltaP_prev_grid[tuple(self.indices.T)] = delta_p_prev_interp.reshape(delta_p_prev_interp.shape[0])

		## Block extraction
		x_list = []
		obst_list = []
		y_list = []
		indices_list = []

		overlap = self.overlap
		shape = self.shape

		n_x = int(np.ceil((grid.shape[2]-shape)/(shape - overlap )) )
		n_y = int((grid.shape[1]-shape)/(shape - overlap ))

		for i in range ( n_y + 2 ): #+1 b
			for j in range ( n_x +1 ):
				
				# going right to left
				x_0 = grid.shape[2] - j*shape + j*overlap - shape
				if j == n_x: x_0 = 0
				x_f = x_0 + shape

				y_0 = i*shape - i*overlap
				if i == n_y + 1: y_0 = grid.shape[1]-shape
				y_f = y_0 + shape

				x_list.append(grid[0:1, y_0:y_f, x_0:x_f, 0:4])
				y_list.append(grid[0:1, y_0:y_f, x_0:x_f, 4:5])

				indices_list.append([i, n_x - j])

		self.x_array = np.concatenate(x_list)
		self.y_array = np.concatenate(y_list)

		y_array = self.y_array
		N = self.x_array.shape[0]
		features = self.x_array.shape[3]
		
		for step in range(y_array.shape[0]):
			y_array[step,...,0][self.x_array[step,...,3] != 0] -= np.mean(y_array[step,...,0][self.x_array[step,...,3] != 0])

		x_array_flat = self.x_array.reshape((N, self.x_array.shape[1]*self.x_array.shape[2], features ))
		input_flat = x_array_flat.reshape((x_array_flat.shape[0],-1))

		input_transformed = self.pcainput.transform(input_flat)[:,:self.pc_in]
		print(' Total variance from input represented: ' + str(np.sum(self.pcainput.explained_variance_ratio_[:self.pc_in])))
		print(input_transformed.shape)

		# to with label data
		y_array_flat = y_array.reshape((N, y_array.shape[1]*y_array.shape[2], 1))
		y_array_flat = y_array_flat.reshape((y_array_flat.shape[0],-1))

		y_transformed = self.pcap.transform(y_array_flat)[:,:self.pc_p]
		print(' Total variance from pressure represented: ' + str(np.sum(self.pcap.explained_variance_ratio_[:self.pc_p])))

		if self.standardization_method == 'std':
			## Option 1: Standardization
			data = np.load('mean_std.npz')
			mean_in_loaded = data['mean_in']
			std_in_loaded = data['std_in']
			mean_out_loaded = data['mean_out']
			std_out_loaded = data['std_out']
			x_input = (input_transformed - mean_in_loaded) / std_in_loaded
		elif self.standardization_method == 'min_max':
			## Option 2: Min-max scaling
			data = np.load('min_max_values.npz')
			min_in_loaded = data['min_in']
			max_in_loaded = data['max_in']
			min_out_loaded = data['min_out']
			max_out_loaded = data['max_out']
			x_input = (input_transformed - min_in_loaded) / (max_in_loaded - min_in_loaded)
		elif self.standardization_method == 'max_abs':
			## Option 3: Old method
			x_input = input_transformed / self.max_abs_input_PCA
		else:
			raise ValueError("Standardization method not valid")

		comp = self.pcap.components_
		pca_mean = self.pcap.mean_

		res_concat = np.array(self.model(np.array(x_input)))

		if self.standardization_method == 'std':
			res_concat = (res_concat * std_out_loaded) + mean_out_loaded
		elif self.standardization_method == 'min_max':
			res_concat = res_concat * (max_out_loaded - min_out_loaded) + min_out_loaded
		elif self.standardization_method == 'max_abs':
			res_concat *= self.max_abs_output_PCA
		else:
			raise ValueError("Standardization method not valid")

		res_flat_inv = np.dot(res_concat, comp[:self.pc_p, :]) + pca_mean
		res_concat = res_flat_inv.reshape((res_concat.shape[0], shape, shape, 1))

		# to test with label data
		y_flat_inv = np.dot(y_transformed, comp[:self.pc_p, :]) + pca_mean	
		y_concat = y_flat_inv.reshape((res_concat.shape[0], shape, shape, 1)) 

		## Dimensionalize pressure field - There is no need to dimensionalize the pressure field here
		## As we can compare it to the reference non-dimensionalized field
		## This only needs to be done when calling the SM in the CFD solver
		res_concat = res_concat  * self.max_abs_delta_p * pow(U, 2.0)

		#### This gives worse results... #####
		# Ignore blocks with near zero delta_U
		# Assign deltap = 0
		# for i, x in enumerate(x_list):
		# 	if (x[0,:,:,0] < 1e-5).all() and (x[0,:,:,1] < 1e-5).all():
		# 		res_concat[i] = np.zeros((shape, shape, 1))
		#### This gives worse results... #####
		
		# the boundary condition is a fixed pressure of 0 at the output
		self.Ref_BC = 0 

		# performing the assembly process
		apply_deltaU_change_wgt = True
		deltap_res, change_in_deltap = self.assemble_prediction(res_concat[...,0], indices_list, n_x, n_y, apply_filter,
									grid.shape[2], grid.shape[1], deltaU_change_grid, deltaP_prev_grid, apply_deltaU_change_wgt)
		
		# The next line can be used to evaluate the assembly algorithm
		#deltap_test_res = self.assemble_prediction(y_array[...,0], indices_list, n_x, n_y, apply_filter, grid.shape[2], grid.shape[1])
		
		################## ----------------//---------------####################################

		## use field_deltap = deltap_test_res to test the assembly algorith -> it should be almost perfect in that case
		if not apply_deltaU_change_wgt:
			# option 1: use pure deltap
			field_deltap = deltap_res
		else:
			# option 2: use the change in deltap
			field_deltap = deltaP_prev_grid + change_in_deltap
			
		cfd_results = grid[0,:,:,4] * self.max_abs_delta_p * pow(U, 2.0)
		no_flow_bool = grid[0,:,:,3] == 0

		# Plotting the integrated pressure field
		if save_plots or show_plots:
			fig, axs = plt.subplots(3,2, figsize=(65, 15))

			vmax = np.max(cfd_results)
			vmin = np.min(cfd_results)

			masked_arr = np.ma.array(field_deltap, mask=no_flow_bool)
			axs[0,0].set_title('delta_p predicted', fontsize = 15)
			cf = axs[0,0].imshow(masked_arr, interpolation='nearest', cmap='viridis')#, vmax = vmax, vmin = vmin )
			plt.colorbar(cf, ax=axs[0,0])

			masked_arr = np.ma.array(cfd_results, mask=no_flow_bool)
			axs[1,0].set_title('CFD results', fontsize = 15)
			cf = axs[1,0].imshow(masked_arr, interpolation='nearest', cmap='viridis')#, vmax = vmax, vmin = vmin)
			plt.colorbar(cf, ax=axs[1,0])

			masked_arr = np.ma.array( np.abs(( cfd_results - field_deltap )/(np.max(cfd_results) -np.min(cfd_results))*100) , mask=no_flow_bool)
			axs[2,0].set_title('error in %', fontsize = 15)
			cf = axs[2,0].imshow(masked_arr, interpolation='nearest', cmap='viridis', vmax = 10, vmin=0 )
			plt.colorbar(cf, ax=axs[2,0])

			# deltaP values without weighting
			masked_arr = np.ma.array(deltap_res, mask=no_flow_bool)
			axs[0,1].set_title('delta_p predicted - no weighting', fontsize = 15)
			cf = axs[0,1].imshow(masked_arr, interpolation='nearest', cmap='viridis')#, vmax = vmax, vmin = vmin )
			plt.colorbar(cf, ax=axs[0,1])

			masked_arr = np.ma.array(cfd_results, mask=no_flow_bool)
			axs[1,1].set_title('CFD results', fontsize = 15)
			cf = axs[1,1].imshow(masked_arr, interpolation='nearest', cmap='viridis')#, vmax = vmax, vmin = vmin)
			plt.colorbar(cf, ax=axs[1,1])

			masked_arr = np.ma.array( np.abs(( cfd_results - deltap_res )/(np.max(cfd_results) -np.min(cfd_results))*100) , mask=no_flow_bool)
			axs[2,1].set_title('error in %', fontsize = 15)
			cf = axs[2,1].imshow(masked_arr, interpolation='nearest', cmap='viridis', vmax = 10, vmin=0 )
			plt.colorbar(cf, ax=axs[2,1])

		if show_plots:
		        plt.show()

		if save_plots:
		        plt.savefig(f'plots/sim{sim}/deltap_pred_t{time}.png')

		plt.close()

                # Plotting the integrated pressure field
		fig, axs = plt.subplots(3,1, figsize=(30, 15))

		# actual pressure fields

		# Infering p_t-1 from ref p and delta_p
		## grid[...,5] is p without being normalized to [0,1]
		## grid[...,4] is deltaP and was normalized ...

		p_prev = grid[0,:,:,5] - cfd_results
		p_pred = p_prev + field_deltap

		if save_plots or show_plots:
			# Plotting the integrated pressure field
			fig, axs = plt.subplots(3,1, figsize=(30, 15))

			masked_arr = np.ma.array(p_pred, mask=no_flow_bool)
			axs[0].set_title(r'Predicted pressure $p_{t-1} + delta_p$', fontsize = 15)
			cf = axs[0].imshow(masked_arr, interpolation='nearest', cmap='viridis')#, vmax = vmax, vmin = vmin )
			plt.colorbar(cf, ax=axs[0])

			masked_arr = np.ma.array(grid[0,:,:,5], mask=no_flow_bool)
			axs[1].set_title('Pressure (CFD)', fontsize = 15)
			cf = axs[1].imshow(masked_arr, interpolation='nearest', cmap='viridis')#, vmax = vmax, vmin = vmin)
			plt.colorbar(cf, ax=axs[1])

			masked_arr = np.ma.array( np.abs(( grid[0,:,:,5] - p_pred )/(np.max(grid[0,:,:,5]) -np.min(grid[0,:,:,5]))*100) , mask=no_flow_bool)

			axs[2].set_title('error in %', fontsize = 15)
			cf = axs[2].imshow(masked_arr, interpolation='nearest', cmap='viridis', vmax = 2, vmin=0 )
			plt.colorbar(cf, ax=axs[2])

		if show_plots:
			plt.show()

		if save_plots:
			plt.savefig(f'plots/sim{sim}/p_pred_t{time}.png')

		plt.close()

		if save_plots or show_plots:
			# # Plotting the input fields - for debugging purposes
			fig, axs = plt.subplots(4,1, figsize=(65, 15))

			masked_arr = np.ma.array(grid[0,:,:,0], mask=no_flow_bool)
			cf = axs[0].imshow(masked_arr, interpolation='nearest', cmap='viridis')#, vmax = vmax, vmin = vmin )
			plt.colorbar(cf, ax=axs[0])

			masked_arr = np.ma.array(grid[0,:,:,1], mask=no_flow_bool)
			cf = axs[1].imshow(masked_arr, interpolation='nearest', cmap='viridis')#, vmax = vmax, vmin = vmin)
			plt.colorbar(cf, ax=axs[1])

			masked_arr = np.ma.array( grid[0,:,:,2] , mask=no_flow_bool)
			cf = axs[2].imshow(masked_arr, interpolation='nearest', cmap='viridis', vmax = 10, vmin=0 )
			plt.colorbar(cf, ax=axs[2])

			masked_arr = np.ma.array( grid[0,:,:,3] , mask=no_flow_bool)
			cf = axs[2].imshow(masked_arr, interpolation='nearest', cmap='viridis', vmax = 10, vmin=0 )
			plt.colorbar(cf, ax=axs[2])
		
		if save_plots:
			plt.savefig(f'plots/sim{sim}/inputs{sim}.png')
		############## ------------------//------------------##############################
		
		true_masked = cfd_results[~no_flow_bool]
		pred_masked = field_deltap[~no_flow_bool]

		# Calculate norm based on reference data a predicted data
		norm_true = np.max(true_masked) - np.min(true_masked)
		norm_pred = np.max(pred_masked) - np.min(pred_masked)

		# Using the largest norm value to avoid problems with near-zero norms leading to relative errors tending to infinity
		# This bounds the relative errors to [-100%, 100%]
		norm = norm_true #max(norm_true, norm_pred)

		print(f"""
		norm_true = {norm_true};
		norm_pred = {norm_pred};
		""")

		mask_nan = ~np.isnan( pred_masked  - true_masked )

		BIAS_norm = np.mean( (pred_masked  - true_masked )[mask_nan] )/norm * 100
		RMSE_norm = np.sqrt(np.mean( ( pred_masked  - true_masked )[mask_nan]**2 ))/norm * 100
		STDE_norm = np.sqrt( (RMSE_norm**2 - BIAS_norm**2) )
			
		print(f"""
		** Error in delta_p **

		normVal  = {norm} Pa
		biasNorm = {BIAS_norm:.3f}%
		stdeNorm = {STDE_norm:.3f}%
		rmseNorm = {RMSE_norm:.3f}%
		""", flush = True)

		self.pred_minus_true.append( np.mean( (pred_masked  - true_masked )[mask_nan] )/norm )
		self.pred_minus_true_squared.append( np.mean( (pred_masked  - true_masked )[mask_nan]**2 )/norm**2 )
		
		## Error in crude deltaP - withouth weighting
		pred_masked = deltap_res[~no_flow_bool]

		norm_pred = np.max(pred_masked) - np.min(pred_masked)
		norm = norm_true #max(norm_true, norm_pred)

		BIAS_norm = np.mean( (pred_masked  - true_masked )[mask_nan] )/norm * 100
		RMSE_norm = np.sqrt(np.mean( ( pred_masked  - true_masked )[mask_nan]**2 ))/norm * 100
		STDE_norm = np.sqrt( (RMSE_norm**2 - BIAS_norm**2) )

		print(f"""
		** Error in delta_p - no weighting **

		normVal  = {norm} Pa
		biasNorm = {BIAS_norm:.3f}%
		stdeNorm = {STDE_norm:.3f}%
		rmseNorm = {RMSE_norm:.3f}%
		""", flush = True)

		self.pred_minus_true_deltap_crude.append( np.mean( (pred_masked  - true_masked )[mask_nan] )/norm )
		self.pred_minus_true_squared_deltap_crude.append( np.mean( (pred_masked  - true_masked )[mask_nan]**2 )/norm**2 )

		## Error in p

		true_masked = grid[0,:,:,5][~no_flow_bool]
		pred_masked = p_pred[~no_flow_bool]

		norm_true = np.max(true_masked) - np.min(true_masked)
		norm_pred = np.max(pred_masked) - np.min(pred_masked)
		norm = norm_true #max(norm_true, norm_pred)

		mask_nan = ~np.isnan( pred_masked  - true_masked )

		BIAS_norm = np.mean( (pred_masked  - true_masked )[mask_nan] )/norm * 100
		RMSE_norm = np.sqrt(np.mean( ( pred_masked  - true_masked )[mask_nan]**2 ))/norm * 100
		STDE_norm = np.sqrt( (RMSE_norm**2 - BIAS_norm**2) )

		print(f"""
		** Error in p **

		normVal  = {norm} Pa
		biasNorm = {BIAS_norm:.5f}%
		stdeNorm = {STDE_norm:.5f}%
		rmseNorm = {RMSE_norm:.5f}%
		""", flush = True)

		self.pred_minus_true_p.append( np.mean( (pred_masked  - true_masked )[mask_nan] )/norm )
		self.pred_minus_true_squared_p.append( np.mean( (pred_masked  - true_masked )[mask_nan]**2 )/norm**2 )


		return 0

	def createGIF(self, n_sims, n_ts):
		
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



def call_SM_main(delta, model_name, shape, overlap_ratio, var_p, var_in, max_num_PC, dataset_path, \
					plot_intermediate_fields, standardization_method, k, save_plots, show_plots, apply_filter, create_GIF, \
					n_sims, n_ts, phis_fn):

	if save_plots:
		path = 'plots/'
		if os.path.exists(path):
			shutil.rmtree(path)
		os.makedirs(path)

	overlap = int(overlap_ratio*shape)
	
	Eval = Evaluation(delta, shape, overlap, var_p, var_in, dataset_path, model_name, max_num_PC, standardization_method, k, phis_fn)

	Eval.pred_minus_true = []
	Eval.pred_minus_true_squared = []

	Eval.pred_minus_true_deltap_crude = []
	Eval.pred_minus_true_squared_deltap_crude = []

	Eval.pred_minus_true_p = []
	Eval.pred_minus_true_squared_p = []

	# Simulations to use for evaluation
	# This points to the number of the simulation data in the dataset
	sims = list(range(n_sims))
	phi_list = np.loadtxt(phis_fn, dtype=float)

	for sim in sims:
		sim+=1
		path = f'plots/sim{sim}'
		if os.path.exists(path):
			shutil.rmtree(path)
		os.makedirs(path)
		Eval.computeOnlyOnce(sim)
		# Reading sim properties
		phi = phi_list[sim]
		print(f'phi {phi}')
		# Timesteps used for evaluation
		for time in range(n_ts):
			time+=16
			Eval.timeStep(sim, time, plot_intermediate_fields, save_plots, show_plots, apply_filter, phi)

		BIAS_value_deltap_crude = np.mean(Eval.pred_minus_true_deltap_crude[-n_ts:]) * 100
		RMSE_value_deltap_crude = np.sqrt(np.mean(Eval.pred_minus_true_squared_deltap_crude[-n_ts:])) * 100
		STDE_value_deltap_crude = np.sqrt( RMSE_value_deltap_crude**2 - BIAS_value_deltap_crude**2 )

		BIAS_value = np.mean(Eval.pred_minus_true[-n_ts:]) * 100
		RMSE_value = np.sqrt(np.mean(Eval.pred_minus_true_squared[-n_ts:])) * 100
		STDE_value = np.sqrt( RMSE_value**2 - BIAS_value**2 )
		print(f'''
		        Average in SIM {sim}:

		        ** Error in delta_p **

		        BIAS: {BIAS_value:.3f}%
		        STDE: {STDE_value:.3f}%
		        RMSE: {RMSE_value:.3f}%

			** Error in delta_p - no weighting **

		        BIAS: {BIAS_value_deltap_crude:.3f}%
		        STDE: {STDE_value_deltap_crude:.3f}%
		        RMSE: {RMSE_value_deltap_crude:.3f}%
			''')


	BIAS_value = np.mean(Eval.pred_minus_true) * 100
	RMSE_value = np.sqrt(np.mean(Eval.pred_minus_true_squared)) * 100
	STDE_value = np.sqrt( RMSE_value**2 - BIAS_value**2 )

	BIAS_value_deltap_crude = np.mean(Eval.pred_minus_true_deltap_crude) * 100
	RMSE_value_deltap_crude = np.sqrt(np.mean(Eval.pred_minus_true_squared_deltap_crude)) * 100
	STDE_value_deltap_crude = np.sqrt( RMSE_value**2 - BIAS_value**2 )

	BIAS_value_p = np.mean(Eval.pred_minus_true_p) * 100
	RMSE_value_p = np.sqrt(np.mean(Eval.pred_minus_true_squared_p)) * 100
	STDE_value_p = np.sqrt( RMSE_value_p**2 - BIAS_value_p**2 )

	print(f'''
	Average across the WHOLE set of simulations:

	** Error in delta_p **

	BIAS: {BIAS_value:.3f}%
	STDE: {STDE_value:.3f}%
	RMSE: {RMSE_value:.3f}%

	** Error in delta_p - w/ weighting**

	BIAS: {BIAS_value_deltap_crude:.3f}%
	STDE: {STDE_value_deltap_crude:.3f}%
	RMSE: {RMSE_value_deltap_crude:.3f}%

	** Error in p **
	BIAS: {BIAS_value_p:.5f}%
	STDE: {STDE_value_p:.5f}%
	RMSE: {RMSE_value_p:.5f}%
	''', flush = True)

	if create_GIF:
		self.createGIF(n_sims, n_ts)


if __name__ == '__main__':

	delta = 5e-3
	model_name = 'model_small-std-0.95.h5'
	shape = 128
	overlap_ratio = 0.25
	var_p = 0.95
	var_in = 0.95
	max_num_PC = 128
	dataset_path = '../dataset_plate_deltas_5sim20t.hdf5' #adjust dataset path
	standardization_method = 'std'

	plot_intermediate_fields = True
	save_plots = True
	show_plots = False
	apply_filter = False
	create_GIF = True

	n_sims = 5
	n_ts = 5

	call_SM_main(delta, model_name, shape, overlap_ratio, var_p, var_in, max_num_PC, dataset_path,	\
				plot_intermediate_fields, standardization_method, save_plots, show_plots, apply_filter, create_GIF, \
				n_sims, n_ts)

