
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
import math
import scipy.spatial.qhull as qhull

import pickle as pk
import itertools
from scipy.spatial import cKDTree as KDTree
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.path as mpltPath

from shapely.geometry import MultiPoint
from scipy.spatial import distance
import scipy
import scipy.ndimage as ndimage


class Evaluation():
	def __init__(self, delta, shape, avance, var_p, var_in, hdf5_path, model_directory):
		self.delta = delta
		self.shape = shape
		self.avance = avance
		self.var_p = var_in
		self.var_p = var_p
		self.hdf5_path = hdf5_path

		maxs = np.loadtxt('maxs')
		maxs_PCA = np.loadtxt('maxs_PCA')

		self.max_abs_Ux, self.max_abs_Uy, self.max_abs_dist, self.max_abs_p = maxs[0], maxs[1], maxs[2], maxs[3]
		self.max_abs_input_PCA, self.max_abs_p_PCA = maxs_PCA[0], maxs_PCA[1]

		#### loading the model #######
		self.model = tf.keras.models.load_model(model_directory)

		### loading the pca matrices for transformations ###
		self.pcainput = pk.load(open("ipca_input_more.pkl",'rb'))
		self.pcap = pk.load(open("ipca_p_more.pkl",'rb'))

		self.pc_p = np.argmax(self.pcap.explained_variance_ratio_.cumsum() > 0.95) if np.argmax(self.pcap.explained_variance_ratio_.cumsum() > 0.95) > 1 and np.argmax(self.pcap.explained_variance_ratio_.cumsum() > 0.95) <= 64 else 64  #max defined to be 32 here
		self.pc_in = np.argmax(self.pcainput.explained_variance_ratio_.cumsum() > 0.995) if np.argmax(self.pcainput.explained_variance_ratio_.cumsum() > 0.995) > 1 and np.argmax(self.pcainput.explained_variance_ratio_.cumsum() > 0.995) <= 64 else 64


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

	def interpolate_fill(self, values, vtx, wts, fill_value=np.nan):
		ret = np.einsum('nj,nj->n', np.take(values, vtx), wts)
		ret[np.any(wts < 0, axis=1)] = fill_value
		return ret

	
	def create_uniform_grid(self, x_min, x_max,y_min,y_max): #creates a uniform quadrangular grid envolving every cell of the mesh

		X0 = np.linspace(x_min + self.delta/2 , x_max - self.delta/2 , num = int(round( (x_max - x_min)/self.delta )) )
		Y0 = np.linspace(y_min + self.delta/2 , y_max - self.delta/2 , num = int(round( (y_max - y_min)/self.delta )) )

		XX0, YY0 = np.meshgrid(X0,Y0)
		return XX0.flatten(), YY0.flatten()

	#@njit(nopython = True)  #much faster using numba.njit but is giving a bug
	def index(self, array, item):
		for idx, val in np.ndenumerate(array):
			if val == item:
				return idx
			# else:
			# 	return None
		# If no item was found return None, other return types might be a problem due to
		# numbas type inference.

	def read_dataset(self, path, sim , time):

		hdf5_file = h5py.File(path, "r")
		data = hdf5_file["sim_data"][sim:sim+1,time:time+1, ...]
		top_boundary = hdf5_file["top_bound"][sim:sim+1, time:time+1 , ...]
		obst_boundary = hdf5_file["obst_bound"][sim:sim+1, time:time+1 , ...]
		hdf5_file.close()
		return data, top_boundary, obst_boundary

	def computeOnlyOnce(self, sim):

		time = 0
		data, top_boundary, obst_boundary = self.read_dataset(self.hdf5_path, sim , time)

		#arrange data in array:

		i = 0
		self.indice = self.index(data[i,0,:,0] , -100.0 )[0]

		x_min = round(np.min(data[i,0,...,:self.indice,3]),2) 
		x_max = round(np.max(data[i,0,...,:self.indice,3]),2) 

		y_min = round(np.min(data[i,0,...,:self.indice,4]),2)  #- 0.3
		y_max = round(np.max(data[i,0,...,:self.indice,4]),2)  #+ 0.3

		######### -------------------- Assuming constant mesh, the following can be done out of the for cycle ------------------------------- ##########

		X0, Y0 = self.create_uniform_grid(x_min, x_max, y_min, y_max)
		xy0 = np.concatenate((np.expand_dims(X0, axis=1),np.expand_dims(Y0, axis=1)), axis=-1)
		points = data[i,0,:self.indice,3:5] #coordinates

		self.vert, self.weights = self.interp_weights(points, xy0) #takes ~100% of the interpolation cost and it's only done once for each different mesh/simulation case

		# boundaries indice
		indice_top = self.index(top_boundary[i,0,:,0] , -100.0 )[0]
		top = top_boundary[i,0,:indice_top,:]
		max_x, max_y, min_x, min_y = np.max(top[:,0]), np.max(top[:,1]) , np.min(top[:,0]) , np.min(top[:,1])

		is_inside_domain = ( xy0[:,0] <= max_x)  * ( xy0[:,0] >= min_x ) * ( xy0[:,1] <= max_y ) * ( xy0[:,1] >= min_y ) #rhis is just for simplification

		indice_obst = self.index(obst_boundary[i,0,:,0] , -100.0 )[0]
		obst = obst_boundary[i,0,:indice_obst,:]

		obst_points =  MultiPoint(obst)

		hull = obst_points.convex_hull       #only works for convex geometries
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
		obst_bool  = np.zeros((self.grid_shape_y,self.grid_shape_x,1))
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



	def timeStep(self, sim, time, save_plots, show_plots, apply_filter):

			data, top_boundary, obst_boundary = self.read_dataset(self.hdf5_path, sim , time)
			i = 0
			j = 0

			p = data[i,j,:self.indice,2:3] #values
			Ux = data[i,j,:self.indice,0:1] #values
			Uy = data[i,j,:self.indice,1:2] #values

			U_max_norm = np.max(np.sqrt(np.square(Ux) + np.square(Uy))) 

			p_adim = p/pow(U_max_norm,2.0) 
			Ux_adim = Ux/U_max_norm 
			Uy_adim = Uy/U_max_norm 

			p_interp = self.interpolate_fill(p_adim, self.vert, self.weights) #compared to the griddata interpolation 
			Ux_interp = self.interpolate_fill(Ux_adim, self.vert, self.weights)#takes virtually no time  because "vert" and "weigths" where already calculated
			Uy_interp = self.interpolate_fill(Uy_adim, self.vert, self.weights)

			grid = np.zeros(shape=(1, self.grid_shape_y, self.grid_shape_x, 4))

			grid[0,:,:,0:1][tuple(self.indices.T)] = Ux_interp.reshape(Ux_interp.shape[0],1)
			grid[0,:,:,1:2][tuple(self.indices.T)] = Uy_interp.reshape(Uy_interp.shape[0],1)
			grid[0,:,:,2:3] = self.sdfunct
			grid[0,:,:,3:4][tuple(self.indices.T)] = p_interp.reshape(p_interp.shape[0],1)

			grid[np.isnan(grid)] = 0 #set any nan value to 0

			grid[0,:,:,0:1] = grid[0,:,:,0:1]/self.max_abs_Ux
			grid[0,:,:,1:2] = grid[0,:,:,1:2]/self.max_abs_Uy
			grid[0,:,:,2:3] = grid[0,:,:,2:3]/self.max_abs_dist
			grid[0,:,:,3:4] = grid[0,:,:,3:4]/self.max_abs_p

		#	plt.imshow(grid[0,:,:,2] != 0 )
		#	plt.show()

			#create data to pass in the model:
			x_list = []
			obst_list = []
			y_list = []
			indices_list = []

			avance = self.avance
			shape = self.shape

			n_x = int((grid.shape[2]-shape)/(shape - avance ))  
			n_y = int((grid.shape[1]-shape)/(shape - avance ))


			for i in range ( n_y + 2 ): #+1 b
				for j in range ( n_x +1 ):

					if i == (n_y + 1 ):
						x_list.append(grid[0:1, (grid.shape[1]-shape):grid.shape[1] , ((grid.shape[2]-shape)-j*shape+j*avance):(grid.shape[2]-j*shape +j*avance) ,0:3])
						indices_list.append([i, n_x - j  ])
						y_list.append(grid[0:1, (grid.shape[1]-shape):grid.shape[1] , ((grid.shape[2]-shape)-j*shape+j*avance):(grid.shape[2]-j*shape +j*avance) ,3:4])
						#print(int((grid.shape[2]-shape)/(shape - avance )) - j)
						#print([ (grid.shape[1]-shape),grid.shape[1] , ((grid.shape[2]-shape)-j*shape+j*avance),(grid.shape[2]-j*shape +j*avance) ])

					else:
						x_list.append(grid[0:1,(i*shape - i*avance):(shape*(i+1) - i*avance),((grid.shape[2]-shape)-j*shape+j*avance):(grid.shape[2]-j*shape +j*avance),0:3])
						#print([ (i*shape - i*avance) , (shape*(i+1) - i*avance) , ((grid.shape[2]-shape)-j*shape+j*avance) , (grid.shape[2]-j*shape +j*avance) ] )
						indices_list.append([i, n_x - j]) #will be used to rearrange the output
						y_list.append(grid[0:1,(i*shape - i*avance):(shape*(i+1) - i*avance),((grid.shape[2]-shape)-j*shape+j*avance):(grid.shape[2]-j*shape +j*avance),3:4])

						#print(n_x - j)

					if ( j ==  n_x ) and ( i == (n_y+1) ): #last one
						x_list.append(grid[0:1, (grid.shape[1]-shape):grid.shape[1] , 0:shape ,0:3])
						indices_list.append([i,-1])
						y_list.append(grid[0:1, (grid.shape[1]-shape):grid.shape[1] , 0:shape ,3:4])
						#print([ (grid.shape[1]-shape),grid.shape[1], 0,shape] )
						#print(-1)

					elif j == n_x :
						x_list.append(grid[0:1,i*shape - i*avance :shape*(i+1) -i*avance , 0:shape ,0:3])
						indices_list.append([i,-1])
						y_list.append(grid[0:1,i*shape - i*avance :shape*(i+1) -i*avance , 0:shape ,3:4])
						#print([i*shape - i*avance,shape*(i+1) -i*avance, 0,shape])
						#print(-1)

			x_array = np.concatenate(x_list)
			y_array = np.concatenate(y_list)


			N = x_array.shape[0]
			features = x_array.shape[3]

			x_array_flat = x_array.reshape((N, x_array.shape[1]*x_array.shape[2], features ))
			y_array_flat = y_array.reshape((N, y_array.shape[1]*y_array.shape[2] ))
			input_flat = x_array_flat.reshape((x_array_flat.shape[0],-1))

			input_transformed = self.pcainput.transform(input_flat)[:,:self.pc_in]
			print(' Total variance from input represented: ' + str(np.sum(self.pcainput.explained_variance_ratio_[:self.pc_in])))
			print(input_transformed.shape)

			y_transformed = self.pcap.transform(y_array_flat)[:,:self.pc_p]
			print(' Total variance from Obst_bool represented: ' + str(np.sum(self.pcap.explained_variance_ratio_[:self.pc_p])))


			x_input = input_transformed/self.max_abs_input_PCA
		
			comp = self.pcap.components_
			pca_mean = self.pcap.mean_

			result_array = np.empty(grid[...,0:1].shape)

			res_concat = np.array(self.model(np.array(x_input)))

			res_flat_inv = np.dot(res_concat*self.max_abs_p_PCA, comp[:self.pc_p, :]) + pca_mean	
			
			res_concat = res_flat_inv.reshape((res_concat.shape[0], shape, shape, 1))

			#correction

			flow_bool_ant = np.ones((shape,shape))
			BC_up = 0
			BC_ant = 0
			BC_alter = 0

			BC_ups = np.zeros(n_x+1)

			for i in range(len(x_list)):

				idx = indices_list[i]
				flow_bool = x_array[i,:,:,2]

				res = res_concat[i,:,:,0]

				if idx[0] == 0: 
					if idx[1] == n_x :

						BC_coor = np.mean(res[:,(shape-avance):shape][flow_bool[:,(shape-avance):shape]!=0]) - BC_up
						res -= BC_coor
						BC_ups[idx[1]] = np.mean(res[(shape-avance):shape,(shape-avance):shape][flow_bool[(shape-avance):shape,(shape-avance):shape] !=0])	

						
					elif idx[1] == -1:
						p_j = (grid.shape[2]-shape)-n_x*shape+n_x*avance
						BC_coor = np.mean(res[:, p_j:p_j + avance][flow_bool[:, p_j:p_j + avance] !=0] ) - BC_ant_0 #middle ones are corrected by the right side
						res -= BC_coor
						BC_up_ = np.mean(res[(shape-avance):shape, p_j:p_j + avance][flow_bool[(shape-avance):shape, p_j:p_j + avance] !=0] ) #equivale a BC_ups[idx[1]==-1]
					else:
						BC_coor = np.mean(res[:,(shape-avance):shape][flow_bool[:,(shape-avance):shape] !=0] ) - BC_ant_0 
						res -= BC_coor
						BC_ups[idx[1]] = np.mean(res[(shape-avance):shape,:][flow_bool[(shape-avance):shape,:] !=0])	
					BC_ant_0 =  np.mean(res[:,0:avance][flow_bool[:,0:avance] !=0]) 	

				elif idx[0] == n_y+1 : 
					if idx[1] == -1: 
				
						p = grid.shape[1] - (shape*(n_y+1) - n_y*avance)
						p_j = (grid.shape[2]-shape)-n_x*shape+n_x*avance
						BC_coor = np.mean(res[shape - p -avance: shape - p , p_j: p_j + avance][flow_bool[ shape - p -avance: shape - p , p_j: p_j + avance] !=0] ) - BC_up_
						res -= BC_coor
					else: 
				
			#				start_i = (n_y*avance - int(n_y*avance/shape)*shape)
			#				BC_coor = np.mean(res[shape - avance - start_i: shape - start_i,(shape-avance):shape][flow_bool[shape - avance - start_i: shape - start_i,(shape-avance):shape] ==1]) - BC_ups[idx[1]] #o problema pode estar aqui

						p = grid.shape[1] - (shape*(n_y+1) - n_y*avance)
						if np.isnan(BC_ups[idx[1]]):
							BC_coor = np.mean(res[:,shape-avance:shape][flow_bool[:,shape-avance:shape] !=0]) - BC_alter
						else:
							BC_coor = np.mean(res[shape - p -avance: shape - p,:][flow_bool[shape - p -avance: shape - p,:] !=0]) - BC_ups[idx[1]]

			
						res -= BC_coor	

				else:

					if idx[1] == -1:

						p_j = (grid.shape[2]-shape)-n_x*shape+n_x*avance
						BC_coor = np.mean(res[0:avance,p_j: p_j + avance ][flow_bool[0:avance,p_j: p_j + avance ]!=0]) - BC_up_
						res -= BC_coor
						BC_up_ = np.mean(res[(shape-avance):shape, p_j: p_j + avance])

					else:

						if np.isnan(BC_ups[idx[1]]):
							BC_coor = np.mean(res[: ,shape-avance:shape][flow_bool[: ,shape-avance:shape] !=0]) - BC_alter
						else:
							BC_coor = np.mean(res[0:avance,:][flow_bool[0:avance,:]!=0]) - BC_ups[idx[1]]
						res -= BC_coor 
						BC_ups[idx[1]] = np.mean(res[(shape-avance):shape,:][flow_bool[(shape-avance):shape,:] !=0])	
						


				BC_alter = np.mean(res[:,0:avance][flow_bool[:,0:avance] !=0]) #BC alternative : lado direito para quando nao dá +para corrigir por cima




				if idx == [n_y +1, -1]:
					result_array[0,(grid.shape[1]-(shape-avance)):grid.shape[1] , 0:(grid.shape[2] - (n_x+1)*(shape-avance) -avance) ,0] = res[avance:shape , 0:grid.shape[2] - (n_x+1)*(shape-avance) -avance]

					#print([ (grid.shape[1]-shape), grid.shape[1] , 0, shape])

				elif idx[1] == -1:
					#print([idx[0]*shape- idx[0],(1+idx[0])*shape- idx[0], 0,shape])

					result_array[0,(idx[0]*shape - idx[0]*avance):(1+idx[0])*shape - idx[0]*avance, 0:shape,0] = res


				elif idx[0] == (n_y + 1):
					j = n_x - idx[1]

					#print((grid.shape[1]-shape), grid.shape[1], grid.shape[2] -shape - j*(shape-avance) , grid.shape[2] - j*(shape-avance))
					result_array[0,(grid.shape[1]-(shape-avance)):grid.shape[1], grid.shape[2] -shape - j*(shape-avance) : grid.shape[2] - j*(shape-avance) ,0] = res[avance:shape,:]

				else:

					j = n_x - idx[1]

					#print((idx[0]*shape - idx[0]*avance),(1+idx[0])*shape - idx[0]*avance, grid.shape[2] -shape - j*(shape-avance) , grid.shape[2] - j*(shape-avance))

					result_array[0,(idx[0]*shape - idx[0]*avance):(1+idx[0])*shape - idx[0]*avance, grid.shape[2] -shape - j*(shape-avance) : grid.shape[2] - j*(shape-avance) ,0] = res


			result_array -= np.mean( 3* result_array[:,:,-1,:] - result_array[:,:,-2,:] )/3



			################### this applies a gaussian filter to remove boundary artifacts #################
			if apply_filter:

				result_array = ndimage.gaussian_filter(result_array[0,:,:,0], sigma=(10, 10), order=0)

			################## ----------------//---------------####################################

			if save_plots:

				masked_arr = np.ma.array(result_array[0,:,:,0], mask=(grid[0,:,:,2] == 0))
				fig, axs = plt.subplots(3,1, figsize=(65, 15))

				vmax = np.max(grid[0,:,:,3])
				vmin = np.min(grid[0,:,:,3])

				axs[0].set_title('Prediction', fontsize = 15)
				cf = axs[0].imshow(masked_arr, interpolation='nearest', cmap='jet')#, vmax = vmax, vmin = vmin )
				plt.colorbar(cf, ax=axs[0])

				masked_arr = np.ma.array(grid[0,:,:,3], mask=(grid[0,:,:,2] == 0))

				axs[1].set_title('CFD', fontsize = 15)
				cf = axs[1].imshow(masked_arr, interpolation='nearest', cmap='jet')#, vmax = vmax, vmin = vmin)
				plt.colorbar(cf, ax=axs[1])

				masked_arr = np.ma.array( np.abs(( grid[0,:,:,3] -result_array[0,:,:,0] )/(np.max(grid[0,:,:,3]) -np.min(grid[0,:,:,3]))*100) , mask=(grid[0,:,:,2] == 0))

				axs[2].set_title('error in %', fontsize = 15)
				cf = axs[2].imshow(masked_arr, interpolation='nearest', cmap='jet', vmax = 10, vmin=0 )
				plt.colorbar(cf, ax=axs[2])

				if show_plots:
					plt.show()

				plt.savefig('/home/paulo/Desktop/plots/' + str(i) + '.png')
				plt.close()

			############## ------------------//------------------##############################

			true_mask = grid[0,:,:,3][grid[0,:,:,2] != 0]
			pred_mask = result_array[0,:,:,0][grid[0,:,:,2] != 0]
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
	path='/home/paulo/Desktop/plots/'

	try:
		shutil.rmtree(path)
	except OSError as e:
		print ("")

	os.makedirs(path)

	###### inputs ########
	delta = 5e-3
	model_directory = 'model_first_.h5'
	shape = 128
	avance = int(0.75*shape)
	var_p = 0.95
	var_in = 0.995
	#hdf5_path = '' #dataset path
	hdf5_path = '/home/paulo/dataset_unsteadyCil_fu_bound.hdf5' #adjust dataset path

	save_plots = True
	show_plots = True
	apply_filter = False


	Eval = Evaluation(delta, shape, avance, var_p, var_in, hdf5_path, model_directory)
	Eval.pred_minus_true = []
	Eval.pred_minus_true_squared = []

	sims = [ 0, 3, 6, 8 ] #phi 0.5, 0.8, 1.1, 1.4

	for sim in sims:
		
		Eval.computeOnlyOnce(sim)

		for time in range(10):

			Eval.timeStep(sim, time, save_plots, show_plots, apply_filter)

	#np.savetxt('errors', error)

	BIAS_value = np.mean(Eval.pred_minus_true) * 100
	RMSE_value = np.sqrt(np.mean(Eval.pred_minus_true_squared)) * 100

	STDE_value = np.sqrt( RMSE_value**2 - BIAS_value**2 )

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