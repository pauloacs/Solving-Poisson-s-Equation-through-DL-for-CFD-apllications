
from ctypes import py_object
import matplotlib
from numpy.core.defchararray import array
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU') 
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)

from numba import njit
import tensorflow as tf
import os
import shutil
import time
import h5py
import numpy as np
import scipy.spatial.qhull as qhull

import pickle as pk
import itertools
import matplotlib.pyplot as plt
import matplotlib.path as mpltPath

from shapely.geometry import MultiPoint
from scipy.spatial import distance
import scipy
import scipy.ndimage as ndimage
from scipy import interpolate

class Evaluation():
	def __init__(self, delta, shape, avance, var_p, var_in, hdf5_path, model_path, max_number_PC):
		"""
		Initialize Evaluation class. 

        Args:
            delta (float): 
            shape (int): 
            avance (float): 
			var_p (float): 
			var_in (float): 
			hdf5_path (str):
			model_path (str):
        """
		self.delta = delta
		self.shape = shape
		self.avance = avance
		self.var_in = var_in
		self.var_p = var_p
		self.hdf5_path = hdf5_path

		maxs = np.loadtxt('maxs')
		maxs_PCA = np.loadtxt('maxs_PCA')

		self.max_abs_Ux, self.max_abs_Uy, self.max_abs_dist, self.max_abs_dPdx, self.max_abs_dPdy = maxs[0], maxs[1], maxs[2], maxs[3], maxs[4]
		self.max_abs_input_PCA, self.max_abs_output_PCA = maxs_PCA[0], maxs_PCA[1]

		#### loading the model #######
		self.model = tf.keras.models.load_model(model_path)
		print(self.model.summary())
		
		### loading the pca matrices for transformations ###
		self.pcainput = pk.load(open("ipca_input.pkl",'rb'))
		self.pcap = pk.load(open("ipca_p.pkl",'rb'))

		self.pc_p = np.argmax(self.pcap.explained_variance_ratio_.cumsum() > self.var_p) if np.argmax(self.pcap.explained_variance_ratio_.cumsum() > self.var_p) > 1 and np.argmax(self.pcap.explained_variance_ratio_.cumsum() > self.var_p) <= max_number_PC else max_number_PC
		self.pc_in = np.argmax(self.pcainput.explained_variance_ratio_.cumsum() > self.var_in) if np.argmax(self.pcainput.explained_variance_ratio_.cumsum() > self.var_in) > 1 and np.argmax(self.pcainput.explained_variance_ratio_.cumsum() > self.var_in) <= max_number_PC else max_number_PC


	def interp_weights(self, xyz, uvw):
		"""
		Gets the interpolation's verticies and weights from xyz to uvw.

		Args:
			xyz (NDArray): Original array of coordinates.
			uvw (NDArray): Target array of coordinates
		"""
		d = 2 #2d interpolation
		tri = qhull.Delaunay(xyz)
		simplex = tri.find_simplex(uvw)
		vertices = np.take(tri.simplices, simplex, axis=0)
		temp = np.take(tri.transform, simplex, axis=0)
		delta = uvw - temp[:, d]
		bary = np.einsum('njk,nk->nj', temp[:, :d, :], delta)
		return vertices, np.hstack((bary, 1 - bary.sum(axis=1, keepdims=True)))

	def interpolate(self, values, vtx, wts):
		"""
		Interpolate based on previously computed vertices (vtx) and weights (wts).

		Args:
			values (NDArray): Array of values to interpolate.
			vtx (NDArray): Array of interpolation vertices.
			wts (NDArray): Array of interpolation weights.
		"""
		return np.einsum('nj,nj->n', np.take(values, vtx), wts)

	def interpolate_fill(self, values, vtx, wts, fill_value=np.nan):
		"""
		Interpolate based on previously computed vertices (vtx) and weights (wts) and fill.

		Args:
			values (NDArray): Array of values to interpolate.
			vtx (NDArray): Array of interpolation vertices.
			wts (NDArray): Array of interpolation weights.
			fill_value (float): Value used to fill.
		"""
		ret = np.einsum('nj,nj->n', np.take(values, vtx), wts)
		ret[np.any(wts < 0, axis=1)] = fill_value
		return ret

	
	def create_uniform_grid(self, x_min, x_max, y_min, y_max):
		"""
		Creates an uniform 2D grid (should envolve every cell of the mesh).

        Args:
            x_min (float): Self-explanatory.
            x_max (float): Self-explanatory.
            y_min (float): Self-explanatory.
			y_max (float): Self-explanatory.
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
		data, top_boundary, obst_boundary = self.read_dataset(self.hdf5_path, sim , time)

		self.indice = self.index(data[0,0,:,0] , -100.0 )[0]


		x_min = round(np.min(data[0,0,...,:self.indice,3]),2) 
		x_max = round(np.max(data[0,0,...,:self.indice,3]),2) 

		y_min = round(np.min(data[0,0,...,:self.indice,4]),2)  #- 0.3
		y_max = round(np.max(data[0,0,...,:self.indice,4]),2)  #+ 0.3

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
		self.max_x, self.max_y, self.min_x, self.min_y = np.max(top[:,0]), np.max(top[:,1]) , np.min(top[:,0]) , np.min(top[:,1])

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

		top = top[0:top.shape[0]:2,:]   #if this has too many values, using cdist can crash the memmory since it needs to evaluate the distance between ~1M points with thousands of points of top
		obst = obst[0:obst.shape[0]:2,:]

		sdf = np.minimum(distance.cdist(xy0,obst).min(axis=1), distance.cdist(xy0,top).min(axis=1) ) * domain_bool

		div = 1 #parameter defining the sliding window vertical and horizontal displacements

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

		dPdx = data[i,j,:self.indice,5:6] #values
		dPdx_interp = self.interpolate_fill(dPdx, self.vert, self.weights) 

		for (step, x_y) in enumerate(xy0):  
			if domain_bool[step] * (~np.isnan(dPdx_interp[step])) :
				jj = int(round((x_y[...,0] - x0) / dx))
				ii = int(round((x_y[...,1] - y0) / dy))

				indices[step,0] = ii
				indices[step,1] = jj
				self.sdfunct[ii,jj,:] = sdf[step]
				obst_bool[ii,jj,:]  = int(1)

		self.indices = indices.astype(int)

		self.pred_minus_true = []
		self.pred_minus_true_squared = []

		return 0

	def assemble_prediction(self, field, array, indices_list, n_x, n_y, apply_filter, shape_x, shape_y):
		"""
		Reconstructs the flow domain based on squared blocks.

		Args:
			field
			array
			indices_list
			n_x
			n_y
			apply_filter
			shape_x
			shape_y 
		"""
		avance = self.avance
		shape = self.shape
		Ref_BC = self.Ref_BC

		result_array = np.empty(shape=(1, shape_y, shape_x, 1))

		BC_ups = np.zeros(n_x+1)

		p_i = shape_y - (shape * (n_y+1) - n_y*avance)
		p_j = (shape_x-shape) - n_x * (shape-avance)

		result = result_array

		for i in range(self.x_array.shape[0]):

			idx_i, idx_j = indices_list[i]
			flow_bool = self.x_array[i,:,:,2]
			pred_field = array[i,...]

			if idx_i == 0: # first row
				if i == 0:
					if field == 'dp_dx': # using inlet boundary condition
						
						# The column of cells where this reference BC is applied must 
						# be part of the flow domain
						col = 0
						while (flow_bool[:,col]!=0).sum() == 0:
							col += 1
							assert col < shape, f"At least the right-most column of the {shape}x{shape} block must belong \
								to the flow domain, or this won't work. Please change the way you construct the dataset, or preprocess it."

						BC_coor = np.mean(pred_field[:,col][flow_bool[:,col]!=0]) - Ref_BC # setting here the reference BC 

					elif field == 'dp_dy': # using top wall boundary condition
						BC_coor = np.mean(pred_field[1,:][flow_bool[1,:]!=0]) - Ref_BC  # i = 0 sits outside the inclusion zone
				else:
					BC_ant_0 = np.mean(old_pred_field[:,-avance:][flow_bool[:,-avance:] !=0]) 
					BC_coor = np.mean(pred_field[:,:avance][flow_bool[:,:avance]!=0]) - BC_ant_0
				if idx_j == n_x:	
					intersect_zone_limit = avance - p_j
					BC_ant_0 = np.mean(old_pred_field[:,-intersect_zone_limit:][flow_bool[:,-intersect_zone_limit:] !=0]) 
					BC_coor = np.mean(pred_field[:,:intersect_zone_limit][flow_bool[:,:intersect_zone_limit]!=0]) - BC_ant_0
				pred_field -= BC_coor
				BC_ups[idx_j] = np.mean(pred_field[-avance:,:][flow_bool[-avance:,:] !=0])

			elif idx_i != n_y + 1: # middle rows
				if np.isnan(BC_ups[idx_j]):
					if idx_j == n_x:
						intersect_zone_limit = avance - p_j
						BC_ant_0 = np.mean(old_pred_field[:,-intersect_zone_limit:][flow_bool[:,-intersect_zone_limit:] !=0]) 
						BC_coor = np.mean(pred_field[:,:intersect_zone_limit][flow_bool[:,:intersect_zone_limit]!=0]) - BC_ant_0
					else:
						BC_ant_0 = np.mean(old_pred_field[:,-avance:][flow_bool[:,-avance:] !=0]) 
						BC_coor = np.mean(pred_field[:,:avance][flow_bool[:,:avance]!=0]) - BC_ant_0											
				else:
					BC_coor = np.mean(pred_field[:avance,:][flow_bool[:avance,:]!=0]) - BC_ups[idx_j]
				pred_field -= BC_coor
				BC_ups[idx_j] = np.mean(pred_field[-avance:,:][flow_bool[-avance:,:] !=0])
				if idx_i == n_y:
					BC_ups[idx_j] = np.mean(pred_field[-(shape-p_i):,:][flow_bool[-(shape-p_i):,:] !=0])
			
			else: # last row
				if np.isnan(BC_ups[idx_j]):
					if idx_j == n_x:
						intersect_zone_limit = avance - p_j
						BC_ant_0 = np.mean(old_pred_field[:,-intersect_zone_limit:][flow_bool[:,-intersect_zone_limit:] !=0]) 
						BC_coor = np.mean(pred_field[:,:intersect_zone_limit][flow_bool[:,:intersect_zone_limit]!=0]) - BC_ant_0
					else:
						BC_ant_0 = np.mean(old_pred_field[:,-avance:][flow_bool[:,-avance:] !=0]) 
						BC_coor = np.mean(pred_field[:,:avance][flow_bool[:,:avance]!=0]) - BC_ant_0								
				else:
					BC_coor = np.mean(pred_field[-p_i-avance:-p_i,:][flow_bool[-p_i-avance:-p_i,:]!=0]) - BC_ups[idx_j]
				pred_field -= BC_coor		
				
			old_pred_field = pred_field
			
			if [idx_i, idx_j] == [n_y + 1, n_x]:
				result[0,(shape_y-(shape-avance)):shape_y , -intersect_zone_limit: ,0] = pred_field[avance:shape , -intersect_zone_limit:]
				#print([ (shape_y-(shape-avance)),shape_y , shape_x-intersect_zone_limit, shape_x])
			elif idx_j == n_x:
				#print([(idx_i*shape - idx_i*avance),(1+idx_i)*shape - idx_i*avance, shape_x-intersect_zone_limit, shape_x])
				result[0,(idx_i*shape - idx_i*avance):(1+idx_i)*shape - idx_i*avance,-intersect_zone_limit:,0] = pred_field[:,-intersect_zone_limit:]
			elif idx_i == (n_y + 1):
				#print((shape_y-(shape-avance)), shape_y, idx_j*(shape-avance), shape + idx_j*(shape-avance))
				result[0,(shape_y-(shape-avance)):shape_y, idx_j*(shape-avance) : shape + idx_j*(shape-avance) ,0] = pred_field[avance:shape,:]
			else:
				#print(((idx_i*shape - idx_i*avance), (1+idx_i)*shape - idx_i*avance, idx_j*(shape-avance), shape + idx_j*(shape-avance)))
				result[0,(idx_i*shape - idx_i*avance):(1+idx_i)*shape - idx_i*avance, idx_j*(shape-avance) : shape + idx_j*(shape-avance) ,0] = pred_field
			
		if field == 'dp_dx':
			result -= np.mean( 3* result[:,:,0,:] - result[:,:,1,:] )/3
		elif field == 'dp_dy':
			result -= np.mean( 3* result[:,1,:,:] - result[:,2,:,:] )/3


		################### this applies a gaussian filter to remove boundary artifacts #################

		if apply_filter:
			result = ndimage.gaussian_filter(result[0,:,:,0], sigma=(10, 10), order=0)

		return result

	def integrate_field(self, block, xl, yl, direction_x=1, direction_y=1):
		"""
		Integrate {pP/dx, dP/dt} into P(x,y).

		Args:
			block (NDArray): 
			xl (NDArray):
			yl (NDArray):
			direction_x (int): 
			direction_y (int): 
		
		"""

		to_cumsum_dPdx = block[...,0].copy()
		to_cumsum_dPdy = block[...,1].copy()

		# trying to reset the cumulative sum at the obstacle
		#doing it at once was giving problems

		ll = []
		for i in range(to_cumsum_dPdx.shape[0]):
			aaa = to_cumsum_dPdx[i,:].copy()
			ccc = np.cumsum(aaa)#, axis=1)
			nn = self.sdfunct[i,:,0].astype(int)
			dd = np.diff(np.concatenate(([0.], ccc[nn])))
			aaa[nn] = -dd
			SdPx = np.cumsum(aaa)*np.diff(xl)[0]
			SdPx = SdPx.reshape((1, SdPx.shape[0]))
			ll.append(SdPx)
		
		SdPx = np.concatenate(ll, axis =0)
		#SdPx = np.cumsum(test_dPdx[0,:,:,0], axis=1)*np.diff(xl)[0]
		SdPy = np.cumsum(to_cumsum_dPdy, axis=0)*np.diff(yl)[0]

		Phat = np.zeros(SdPx.shape)
		for i in range(Phat.shape[0]):
			for j in range(Phat.shape[1]):
				j *= direction_x
				i *= direction_y
				initial_j = 0 
				initial_i = 0
				if direction_x == -1: initial_j = -1 
				if direction_y == -1: initial_i = -1 
				Phat[i,j] += np.sum([SdPy[i,initial_j], -SdPy[initial_i,initial_j], SdPx[i,j], -SdPx[i,initial_j]])

		return Phat

	def timeStep(self, sim, time, plot_intermediate_fields, save_plots, show_plots, apply_filter):
		"""

		Args:
			sim (int):
			time (int):
			save_plots (bool):
			show_plots (bool):
			apply_filter
		"""
	
		data, top_boundary, obst_boundary = self.read_dataset(self.hdf5_path, sim , time)
		i = 0
		j = 0
		
		Ux = data[i,j,:self.indice,0:1] #values
		Uy = data[i,j,:self.indice,1:2] #values
		p = data[i,j,:self.indice,2:3] #values
		dPdx = data[i,j,:self.indice,6:7] #values
		dPdy = data[i,j,:self.indice,7:8] #values

		U_max_norm = np.max(np.sqrt(np.square(Ux) + np.square(Uy))) 

		dPdx_adim = dPdx * (self.max_x - self.min_x)/pow(U_max_norm,2.0) 
		dPdy_adim = dPdy * (self.max_y - self.min_y)/pow(U_max_norm,2.0) 
		p_adim = p / pow(U_max_norm,2.0) 
		Ux_adim = Ux/U_max_norm 
		Uy_adim = Uy/U_max_norm 

		dPdx_interp = self.interpolate_fill(dPdx_adim, self.vert, self.weights) #compared to the griddata interpolation 
		dPdy_interp = self.interpolate_fill(dPdy_adim, self.vert, self.weights) #compared to the griddata interpolation 
		p_interp = self.interpolate_fill(p_adim, self.vert, self.weights)
		Ux_interp = self.interpolate_fill(Ux_adim, self.vert, self.weights)#takes virtually no time  because "vert" and "weigths" where already calculated
		Uy_interp = self.interpolate_fill(Uy_adim, self.vert, self.weights)

		grid = np.zeros(shape=(1, self.grid_shape_y, self.grid_shape_x, 6))

		grid[0,:,:,0:1][tuple(self.indices.T)] = Ux_interp.reshape(Ux_interp.shape[0],1)
		grid[0,:,:,1:2][tuple(self.indices.T)] = Uy_interp.reshape(Uy_interp.shape[0],1)
		grid[0,:,:,2:3] = self.sdfunct
		grid[0,:,:,3:4][tuple(self.indices.T)] = dPdx_interp.reshape(dPdx_interp.shape[0],1)
		grid[0,:,:,4:5][tuple(self.indices.T)] = dPdy_interp.reshape(dPdy_interp.shape[0],1)
		grid[0,:,:,5:6][tuple(self.indices.T)] = p_interp.reshape(p_interp.shape[0],1)

		grid[np.isnan(grid)] = 0 #set any nan value to 0

		grid[0,:,:,0:1] = grid[0,:,:,0:1]/self.max_abs_Ux
		grid[0,:,:,1:2] = grid[0,:,:,1:2]/self.max_abs_Uy
		grid[0,:,:,2:3] = grid[0,:,:,2:3]/self.max_abs_dist
		grid[0,:,:,3:4] = grid[0,:,:,3:4]/self.max_abs_dPdx
		grid[0,:,:,4:5] = grid[0,:,:,4:5]/self.max_abs_dPdy

		#create data to pass in the model:
		x_list = []
		obst_list = []
		y_list = []
		indices_list = []

		avance = self.avance
		shape = self.shape

		n_x = int(np.ceil((grid.shape[2]-shape)/(shape - avance )) )
		n_y = int((grid.shape[1]-shape)/(shape - avance ))

		# To work with dPdx should start sampling from the left 
		# Because the boundary condition is imposed at the inlet boundary (dp/dx = 0)
		# And afterwards reconstruction must start from the boundary where we know the BC

		for i in range ( n_y + 2 ): #+1 b
			for j in range ( n_x +1 ):

				x_0 = j*shape -j*avance
				if j == n_x: x_0 = grid.shape[2]-shape
				x_f = x_0 + shape

				y_0 = i*shape - i*avance
				if i == n_y + 1: y_0 = grid.shape[1]-shape
				y_f = y_0 + shape

				x_list.append(grid[0:1, y_0:y_f, x_0:x_f, 0:3])
				y_list.append(grid[0:1, y_0:y_f, x_0:x_f, 3:5])

				indices_list.append([i,j])

		self.x_array = np.concatenate(x_list)
		self.y_array = np.concatenate(y_list)
		y_array = self.y_array
		N = self.x_array.shape[0]
		features = self.x_array.shape[3]

		# this is just to check the assembly of the gradients in the physical domain
		for step in range(y_array.shape[0]):
			y_array[step,...,0][self.x_array[step,...,2] != 0] -= np.mean(y_array[step,...,0][self.x_array[step,...,2] != 0])
			y_array[step,...,1][self.x_array[step,...,2] != 0] -= np.mean(y_array[step,...,1][self.x_array[step,...,2] != 0])

		x_array_flat = self.x_array.reshape((N, self.x_array.shape[1]*self.x_array.shape[2], features ))
		y_array_flat = y_array.reshape((N, y_array.shape[1]*y_array.shape[2], 2))
		input_flat = x_array_flat.reshape((x_array_flat.shape[0],-1))
		y_array_flat = y_array_flat.reshape((y_array_flat.shape[0],-1))

		input_transformed = self.pcainput.transform(input_flat)[:,:self.pc_in]
		print(' Total variance from input represented: ' + str(np.sum(self.pcainput.explained_variance_ratio_[:self.pc_in])))
		print(input_transformed.shape)

		y_transformed = self.pcap.transform(y_array_flat)[:,:self.pc_p]
		print(' Total variance from Obst_bool represented: ' + str(np.sum(self.pcap.explained_variance_ratio_[:self.pc_p])))

		x_input = input_transformed/self.max_abs_input_PCA
	
		comp = self.pcap.components_
		pca_mean = self.pcap.mean_

		res_concat = np.array(self.model(np.array(x_input)))
		res_concat *= self.max_abs_output_PCA 

		res_flat_inv = np.dot(res_concat, comp[:self.pc_p, :]) + pca_mean	
		res_concat = res_flat_inv.reshape((res_concat.shape[0], shape, shape, 2)) 

		# Dimensionalize pressure gradient field
		res_concat[...,0] = res_concat[...,0] #* pow(U_max_norm,2.0)/(self.max_x - self.min_x) * self.max_abs_dPdx
		res_concat[...,1] = res_concat[...,1] #* pow(U_max_norm,2.0)/(self.max_y - self.min_y) * self.max_abs_dPdy

		# the boundary condition is 0 for both dP/dx and dP/dy but at different boundaries
		self.Ref_BC = 0 

		# performing the assembly process
		res_dPdx = self.assemble_prediction('dp_dx', res_concat[...,0], indices_list, n_x, n_y, apply_filter, grid.shape[2], grid.shape[1])
		res_dPdy = self.assemble_prediction('dp_dy', res_concat[...,1], indices_list, n_x, n_y, apply_filter, grid.shape[2], grid.shape[1])
		test_dPdx = self.assemble_prediction('dp_dx', y_array[...,0], indices_list, n_x, n_y, apply_filter, grid.shape[2], grid.shape[1])
		test_dPdy = self.assemble_prediction('dp_dy', y_array[...,1], indices_list, n_x, n_y, apply_filter, grid.shape[2], grid.shape[1])
		
		################## ----------------//---------------####################################

		# Plotting intermediate fields: gradP = {dP/dx, dP/dy}

		if plot_intermediate_fields:

			fig, axs = plt.subplots(2,1, figsize=(65, 15))

			masked_arr = np.ma.array(res_dPdx[0,:,:,0], mask=(grid[0,:,:,2] == 0))
			axs[0].set_title('dp/dx predicted', fontsize = 15)
			cf = axs[0].imshow(masked_arr, interpolation='nearest', cmap='jet')#, vmax = vmax, vmin = vmin )
			plt.colorbar(cf, ax=axs[0])

			masked_arr = np.ma.array(test_dPdx[0,:,:,0], mask=(grid[0,:,:,2] == 0))
			axs[1].set_title('Ground truth', fontsize = 15)
			cf = axs[1].imshow(masked_arr, interpolation='nearest', cmap='jet')#, vmax = vmax, vmin = vmin )
			plt.colorbar(cf, ax=axs[1])

			if save_plots:
				fig.savefig('plots/' + 'dp_dx' + str(time) + '.png')

			plt.close()

			fig, axs = plt.subplots(2,1, figsize=(65, 15))
			
			masked_arr = np.ma.array(res_dPdy[0,:,:,0], mask=(grid[0,:,:,2] == 0))
			axs[0].set_title('dp/dy predicted', fontsize = 15)
			cf = axs[0].imshow(masked_arr, interpolation='nearest', cmap='jet')#, vmax = vmax, vmin = vmin )
			plt.colorbar(cf, ax=axs[0])

			masked_arr = np.ma.array(test_dPdy[0,:,:,0], mask=(grid[0,:,:,2] == 0))
			axs[1].set_title('Ground truth', fontsize = 15)
			cf = axs[1].imshow(masked_arr, interpolation='nearest', cmap='jet')#, vmax = vmax, vmin = vmin )
			plt.colorbar(cf, ax=axs[1])

			if show_plots:
				plt.show()

			if save_plots:
				fig.savefig('plots/' + 'dp_dy' + str(time) + '.png')

			plt.close()

		gradP = np.concatenate([res_dPdx, res_dPdy], axis = -1)
		
		xl = np.linspace(self.min_x, self.max_x, grid.shape[2])
		yl = np.linspace(self.min_y, self.max_y, grid.shape[1])

		# Dividing the domain into 4 parts

		center_p_x = int((( xl[self.sdfunct[200,:,0] == 0].max() + xl[self.sdfunct[200,:,0] == 0].min() )/2 - self.X0.min()) / self.delta)
		center_p_y = int(200)

		result = np.empty(gradP[0,:,:,0].shape)

		# block 1 - Upper right
		block_1 = gradP[0,:center_p_y,center_p_x-1:,:].copy()
		mask1 = self.sdfunct[:center_p_y,center_p_x-1,0] != 0 
		pBlock1 = self.integrate_field(block_1, xl, yl, direction_x = -1)
		result[:center_p_y,center_p_x-1:] = pBlock1#[:,::-1]

		# block 2 - Upper left
		block_2 = gradP[0,:center_p_y,:center_p_x,:].copy()
		mask2 = self.sdfunct[:center_p_y,center_p_x,0] != 0 
		pBlock2 = self.integrate_field(block_2, xl, yl)
		pBlock2Corrected = pBlock2 - (pBlock2[:,-1][mask2] - pBlock1[:,0][mask1]).mean() 
		result[:center_p_y,:center_p_x] = pBlock2Corrected

		# block 3 - Lower right
		block_3 = gradP[0,center_p_y:,center_p_x-1:,:].copy()
		mask3 = self.sdfunct[center_p_y:,center_p_x-1,0] != 0 
		pBlock3 = self.integrate_field(block_3, xl, yl, direction_x = -1, direction_y=-1)
		result[center_p_y:,center_p_x-1:] = pBlock3

		# block 4 - Lower left
		block_4 = gradP[0,center_p_y:,:center_p_x,:].copy()
		mask4 = self.sdfunct[center_p_y:,center_p_x,0] != 0 
		pBlock4 = self.integrate_field(block_4, xl, yl, direction_y=-1)
		pBlock4Corrected = pBlock4 - (pBlock4[:,-1][mask4] - pBlock3[:,0][mask3]).mean() 
		result[center_p_y:,:center_p_x] = pBlock4Corrected
	
		field = result 

		# Plotting the integrated pressure field
		masked_arr = np.ma.array(field, mask=(grid[0,:,:,2] == 0))
		fig, axs = plt.subplots(3,1, figsize=(65, 15))

		vmax = np.max(grid[0,:,:,5])
		vmin = np.min(grid[0,:,:,5])

		axs[0].set_title('p predicted', fontsize = 15)
		cf = axs[0].imshow(masked_arr, interpolation='nearest', cmap='jet')#, vmax = vmax, vmin = vmin )
		plt.colorbar(cf, ax=axs[0])

		masked_arr = np.ma.array(grid[0,:,:,5], mask=(grid[0,:,:,2] == 0))

		axs[1].set_title('CFD results', fontsize = 15)
		cf = axs[1].imshow(masked_arr, interpolation='nearest', cmap='jet')#, vmax = vmax, vmin = vmin)
		plt.colorbar(cf, ax=axs[1])

		masked_arr = np.ma.array( np.abs(( grid[0,:,:,5] - field )/(np.max(grid[0,:,:,5]) -np.min(grid[0,:,:,5]))*100) , mask=(grid[0,:,:,2] == 0))

		axs[2].set_title('error in %', fontsize = 15)
		cf = axs[2].imshow(masked_arr, interpolation='nearest', cmap='jet', vmax = 10, vmin=0 )
		plt.colorbar(cf, ax=axs[2])

		if show_plots:
			plt.show()

		if save_plots:
			fig.savefig('plots/' + 'p_pred' + str(time) + '.png')

		plt.close()

		############## ------------------//------------------##############################
		
		# This part is to output the error metrics

		true_mask = grid[0,:,:,3][grid[0,:,:,2] != 0]
		pred_mask = field[grid[0,:,:,2] != 0]
		norm = np.max(grid[0,:,:,3][grid[0,:,:,2] != 0]) - np.min(grid[0,:,:,3][grid[0,:,:,2] != 0])

		mask_nan = ~np.isnan( pred_mask  - true_mask )

		BIAS_norm = np.mean( (pred_mask  - true_mask )[mask_nan] )/norm * 100
		RMSE_norm = np.sqrt(np.mean( ( pred_mask  - true_mask )[mask_nan]**2 ))/norm * 100
		STDE_norm = np.sqrt( (RMSE_norm**2 - BIAS_norm**2) )
		

		# This prints the error metrics for each time frame
		print(f"""
		normVal  = {norm} Pa
		biasNorm = {BIAS_norm:.2f}%
		stdeNorm = {STDE_norm:.2f}%
		rmseNorm = {RMSE_norm:.2f}%
		""")

		self.pred_minus_true.append( np.mean( (pred_mask  - true_mask )[mask_nan] )/norm )
		self.pred_minus_true_squared.append( np.mean( (pred_mask  - true_mask )[mask_nan]**2 )/norm**2 )

		return 0

	
def main():

	### THis creates a directory to save the plots
	path='plots/'

	try:
		shutil.rmtree(path)
	except OSError as e:
		print ("")

	os.makedirs(path)

	###### inputs ########
	delta = 5e-3
	model_directory = 'model_1.h5'
	shape = 128
	avance = int(0.75*shape)
	var_p = 0.95
	var_in = 0.95
	max_number_PC = 512
	hdf5_path = 'dataset_gradP_cil.hdf5' #adjust dataset path

	plot_intermediate_fields = True
	save_plots = True
	show_plots = False
	apply_filter = False

	Eval = Evaluation(delta, shape, avance, var_p, var_in, hdf5_path, model_directory, max_number_PC)
	Eval.pred_minus_true = []
	Eval.pred_minus_true_squared = []

	# Simulations to use for evaluation
	# This point to the number of the simulation data in the dataset
	sims = [0]

	for sim in sims:
		Eval.computeOnlyOnce(sim)
		# Timesteps used for evaluation
		for time in range(5):
			Eval.timeStep(sim, time, plot_intermediate_fields, save_plots, show_plots, apply_filter)

		BIAS_value = np.mean(Eval.pred_minus_true) * 100
		RMSE_value = np.sqrt(np.mean(Eval.pred_minus_true_squared)) * 100
		STDE_value = np.sqrt( RMSE_value**2 - BIAS_value**2 )

		print('Metrics for the whole simulation:')
		print('BIAS for the sim: ' + str(BIAS_value))
		print('RMSE for the sim: ' + str(RMSE_value))
		print('STDE for the sim: ' + str(STDE_value))


if __name__ == '__main__':
	main()

####################### TO CREATE A GIF WITH ALL THE FRAMES ###############################

#filenamesp = []

#for i in range(100):
#  filenamesp.append('/home/paulo/Desktop/plots/' + str(i) +".png") #hardcoded to get the frames in order

#import imageio

#with imageio.get_writer('/home/paulo/Desktop/plots/p_movie.gif', mode='I', duration =0.5) as writer:
#    for filename in filenamesp:
#        image = imageio.imread(filename)
#        writer.append_data(image)

######################## ---------------- //----------------- ###################
