from ctypes import py_object
from re import X
import matplotlib
from numpy.core.defchararray import array
import tensorflow as tf
from tensorflow.python.ops.gen_array_ops import inplace_update
physical_devices = tf.config.list_physical_devices('GPU') 
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)

import dask
import dask.config
import dask.distributed
import dask_ml
import dask_ml.preprocessing
import dask_ml.decomposition

from pyDOE import lhs
from numba import njit
import tensorflow as tf
import os
import shutil
import time
import h5py
import numpy as np
from tensorflow.keras import Model
from tensorflow.keras.layers import ZeroPadding2D, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Conv2DTranspose, concatenate, Input
import math
import scipy.spatial.qhull as qhull
import itertools
from scipy.spatial import cKDTree as KDTree

@njit
def index(array, item):
    for idx, val in np.ndenumerate(array):
        if val == item:
            return idx
    # If no item was found return None, other return types might be a problem due to
    # numbas type inference.

d = 2 #2d interpolation

def interp_weights(xyz, uvw):
    tri = qhull.Delaunay(xyz)
    simplex = tri.find_simplex(uvw)
    vertices = np.take(tri.simplices, simplex, axis=0)
    temp = np.take(tri.transform, simplex, axis=0)
    delta = uvw - temp[:, d]
    bary = np.einsum('njk,nk->nj', temp[:, :d, :], delta)
    return vertices, np.hstack((bary, 1 - bary.sum(axis=1, keepdims=True)))

def interpolate(values, vtx, wts):
    return np.einsum('nj,nj->n', np.take(values, vtx), wts)

def interpolate_fill(values, vtx, wts, fill_value=np.nan):  #this would be the function to fill with nan 
    ret = np.einsum('nj,nj->n', np.take(values, vtx), wts)  #does not work yet
    ret[np.any(wts < 0, axis=1)] = fill_value
    return ret
 
def create_uniform_grid(x_min, x_max,y_min,y_max, delta): #creates a uniform quadrangular grid envolving every cell of the mesh

  X0 = np.linspace(x_min + delta/2 , x_max - delta/2 , num = int(round( (x_max - x_min)/delta )) )
  Y0 = np.linspace(y_min + delta/2 , y_max - delta/2 , num = int(round( (y_max - y_min)/delta )) )

  XX0, YY0 = np.meshgrid(X0,Y0)
  return XX0.flatten(), YY0.flatten()

import matplotlib.pyplot as plt

from scipy.spatial import distance

import matplotlib.path as mpltPath

from shapely.geometry import MultiPoint

from sklearn.decomposition import PCA, IncrementalPCA


import tables

def domain_dist(i,top_boundary, obst_boundary, xy0):

  # boundaries indice
  indice_top = index(top_boundary[i,0,:,0] , -100.0 )[0]
  top = top_boundary[i,0,:indice_top,:]
  max_x, max_y, min_x, min_y = np.max(top[:,0]), np.max(top[:,1]) , np.min(top[:,0]) , np.min(top[:,1])


  #extra_points = np.array( [ ( np.min(top[:,0]) , 0 ), ( np.max(top[:,0]), 0 ) ] )  #to generalize just use top+inlet+ outlet and works for any domain and just remove this
  #top = np.concatenate((top,extra_points), axis=0)  #np.c_[ np.append(top[:,0], extra_points) , np.append(top[:,1], extra_points) ]   

  is_inside_domain = ( xy0[:,0] <= max_x)  * ( xy0[:,0] >= min_x ) * ( xy0[:,1] <= max_y ) * ( xy0[:,1] >= min_y ) #rhis is just for simplification

  # regular polygon for testing

  # # Matplotlib mplPath
  # path = mpltPath.Path(top_inlet_outlet)
  # is_inside_domain = path.contains_points(xy0)
  # print(is_inside_domain.shape)

  indice_obst = index(obst_boundary[i,0,:,0] , -100.0 )[0]
  obst = obst_boundary[i,0,:indice_obst,:]

  obst_points =  MultiPoint(obst)

  hull = obst_points.convex_hull       #only works for convex geometries
  hull_pts = hull.exterior.coords.xy    #have a code for any geometry . enven concave https://stackoverflow.com/questions/14263284/create-non-intersecting-polygon-passing-through-all-given-points/47410079
  hull_pts = np.c_[hull_pts[0], hull_pts[1]]

  path = mpltPath.Path(hull_pts)
  is_inside_obst = path.contains_points(xy0)

  domain_bool = is_inside_domain * ~is_inside_obst

  top = top[0:top.shape[0]:2,:]   #if this has too many values, using cdist can crash the memmory since it needs to evaluate the distance between ~1M points with thousands of points of top
  obst = obst[0:obst.shape[0]:2,:]

  print(top.shape)

  sdf = np.minimum( distance.cdist(xy0,obst).min(axis=1) , distance.cdist(xy0,top).min(axis=1) ) * domain_bool
  print(np.max(distance.cdist(xy0,top).min(axis=1)))
  print(np.max(sdf))

  return domain_bool, sdf

num_sims = 10

def read_dataset(path, print_shape, delta, Nsamples, block_size):

    global num_sims
    x_data = []
    hdf5_file = h5py.File(path, "r")
    data = hdf5_file["sim_data"][:num_sims, :, ...]
    top_boundary = hdf5_file["top_bound"][:num_sims, :, 
    ...]
    obst_boundary = hdf5_file["obst_bound"][:num_sims, :, ...]
    hdf5_file.close()

    filename = 'outarray.h5'
    NUM_COLUMNS = 3

    file = tables.open_file(filename, mode='w')
    atom = tables.Float32Atom()

    array_c = file.create_earray(file.root, 'data', atom, (0, block_size, block_size, NUM_COLUMNS))
    file.close()

    max_abs_fU = []
    max_abs_dist = []
    max_abs_p = []

    for i in range(data.shape[0]):

        indice = index(data[i,0,:,0] , -100.0 )[0]

        #mask_x = np.logical_and(data[i,0,:indice,3] <= 5. , data[i,0,:indice,3] >= -2.)
        data_limited = data[i,0,:indice,:]#[mask_x]

        x_min = round(np.min(data_limited[...,3]),2)
        x_max = round(np.max(data_limited[...,3]),2)

        y_min = round(np.min(data_limited[...,4]),2) #- 0.1
        y_max = round(np.max(data_limited[...,4]),2) #+ 0.1

        delta = delta  # grid resolution --> may be changed

        ######### -------------------- Assuming constant mesh, the following can be done out of the for cycle ------------------------------- ##########

        X0, Y0 = create_uniform_grid(x_min, x_max, y_min, y_max, delta)
        xy0 = np.concatenate((np.expand_dims(X0, axis=1),np.expand_dims(Y0, axis=1)), axis=-1)
        points = data_limited[...,3:5] #coordinates

        vert, weights = interp_weights(points, xy0) #takes ~100% of the interpolation cost and it's only done once for each different mesh/simulation case
    
        domain_bool, sdf = domain_dist(i, top_boundary, obst_boundary, xy0)

        div = 1 #parameter defining the sliding window vertical and horizontal displacements
        
        grid_shape_y = int(round((y_max-y_min)/delta)) 
        grid_shape_x = int(round((x_max-x_min)/delta)) 
        block_size = block_size

        count_ = data.shape[1]* int(grid_shape_y/div - block_size/div + 1 ) * int(grid_shape_x/div - block_size/div + 1 )

        num_samples = int(Nsamples)    #number of samples from each sim -------> needs to be adjusted
        ind_list = np.random.choice(range(count_), num_samples, replace=False)

        count = 0
        cicle = 0

        #arrange data in array: #this can be put outside the j loop if the mesh is constant 

        x0 = np.min(X0)
        y0 = np.min(Y0)
        dx = delta
        dy = delta

        indices= np.zeros((X0.shape[0],2))
        obst_bool  = np.zeros((grid_shape_y,grid_shape_x,1))
        sdfunct = np.zeros((grid_shape_y,grid_shape_x,1))

        p = data_limited[...,2:3] #values
        p_interp = interpolate_fill(p, vert, weights) 
      

        for (step, x_y) in enumerate(xy0):  
            if domain_bool[step] * (~np.isnan(p_interp[step])) :
                jj = int(round((x_y[...,0] - x0) / dx))
                ii = int(round((x_y[...,1] - y0) / dy))

                indices[step,0] = ii
                indices[step,1] = jj
                sdfunct[ii,jj,:] = sdf[step]
                obst_bool[ii,jj,:]  = int(1)

        indices = indices.astype(int)


        for j in range(data.shape[1]):

          data_limited = data[i,j,:indice,:]#[mask_x]


          Ux = data_limited[...,0:1] #values
          Uy = data_limited[...,1:2] #values
          p = data_limited[...,2:3] #values
          fU = data_limited[...,5:6] #values

          U_max_norm = np.max(np.sqrt(np.square(Ux) + np.square(Uy))) 

          p_adim = p/pow(U_max_norm,2.0) 
          fU_adim = fU/pow(U_max_norm,2.0)

          p_interp = interpolate_fill(p_adim, vert, weights) #compared to the griddata interpolation 
          fU_interp = interpolate_fill(fU_adim, vert, weights)#takes virtually no time  because "vert" and "weigths" where already calculated

          #arrange data in array:

          grid = np.zeros(shape=(1, grid_shape_y, grid_shape_x, 4))

          grid[0,:,:,0:1][tuple(indices.T)] = fU_interp.reshape(fU_interp.shape[0],1)
          grid[0,:,:,1:2] = sdfunct
          grid[0,:,:,2:3][tuple(indices.T)] = p_interp.reshape(p_interp.shape[0],1)

          x_list = []
          obst_list = []
          y_list = []


          grid[np.isnan(grid)] = 0 #set any nan value to 0

          lb = np.array([0 + block_size*delta/2 , 0 + block_size*delta/2 ])
          ub = np.array([(x_max-x_min) -block_size*delta/2, (y_max-y_min) - block_size*delta/2])

          XY = lb + (ub-lb)*lhs(2,int(num_samples/100))  #divided by 100 because it samples from each time individually
          XY_indices = (np.round(XY/delta)).astype(int)

          new_XY_indices = [tuple(row) for row in XY_indices]
          XY_indices = np.unique(new_XY_indices, axis=0)

          for [jj, ii] in XY_indices:

                  i_range = [int(ii - block_size/2), int( ii + block_size/2) ]
                  j_range = [int(jj - block_size/2), int( jj + block_size/2) ]

                  x_list.append(grid[0, i_range[0]:i_range[1] , j_range[0]:j_range[1] , 0:1 ])
                  #x_list.append(grid[0, i:(i + 100), j:(j + 100), 0:2 ].transpose(1,0,2))
                  obst_list.append(grid[0, i_range[0]:i_range[1] , j_range[0]:j_range[1] , 1:2 ])
                  # #obst_list.append(grid[0, i:(i + 100), j:(j + 100), 2:3 ].transpose(1,0,2))
                  y_list.append(grid[0, i_range[0]:i_range[1] , j_range[0]:j_range[1] , 2:3 ])
                  # #y_list.append(grid[0, i:(i + 100), j:(j + 100), 3:4 ].transpose(1,0,2))

          cicle += 1
          print(cicle, flush = True)

          x_array = np.array(x_list, dtype = 'float16')#, axis=0 )#.astype('float32') #, dtype = 'float16')
          obst_array = np.array(obst_list, dtype = 'float16')#, axis=0 )#.astype('float16') #np.array(obst_list, dtype = 'float16')
          y_array = np.array(y_list, dtype = 'float16')#, axis=0 )#.astype('float16') #np.array(y_list, dtype = 'float16')

          max_abs_fU.append(np.max(np.abs(x_array[...,0])))
          max_abs_dist.append(np.max(np.abs(obst_array[...,0])))

          for step in range(y_array.shape[0]):
            y_array[step,...][obst_array[step,...] != 0] -= np.mean(y_array[step,...][obst_array[step,...] != 0])
          
          max_abs_p.append(np.max(np.abs(y_array[...,0])))

          array = np.c_[x_array,obst_array,y_array]

          file = tables.open_file(filename, mode='a')
          file.root.data.append(array)
          file.close()

    max_abs_fU = np.max(np.abs(max_abs_fU))
    max_abs_dist = np.max(np.abs(max_abs_dist))
    max_abs_p = np.max(np.abs(max_abs_p))

    print(max_abs_fU, max_abs_dist, max_abs_p)

    np.savetxt('maxs', [max_abs_fU, max_abs_dist, max_abs_p] )

    split = 0.9
    len_train = int(split*x_array.shape[0])
        
    return max_abs_fU, max_abs_dist, max_abs_p, len_train


#maxs = np.loadtxt('maxs') 
#max_abs_fU, max_abs_dist, max_abs_p = maxs[0], maxs[1], maxs[2]

def prepare_data (block_size, Nsamples, num_principal_comp ):

  #load data

  Nsamples = Nsamples
  block_size = block_size

 # client = dask.distributed.Client(processes=False)#, n_workers=16)

  print_shape = True
  max_abs_fU, max_abs_dist, max_abs_p, len_train = read_dataset(hdf5_path, print_shape, delta, Nsamples, block_size)

  ##  PCA Part

  num_principal_comp = num_principal_comp
  square_dim= int(np.sqrt(num_principal_comp))

  filename = 'outarray.h5'

  filename_flat = 'outarray_flat.h5'

  file = tables.open_file(filename_flat, mode='w')
  atom = tables.Float32Atom()

  file.create_earray(file.root, 'data_flat', atom, (0, modes_PCA, 2))
  file.close()

  import pickle as pk

  global ipca_p

  client = dask.distributed.Client(processes=False)#, n_workers=16)

  ipca_p = dask_ml.decomposition.IncrementalPCA(modes_PCA)
  ipca_input = dask_ml.decomposition.IncrementalPCA(modes_PCA)

  N = int(Nsamples * num_sims)

  chunk_size = int(N/300)
  print('Passing the PCA ' + str(N//chunk_size) + ' times', flush = True)

  for i in range(int(N//chunk_size)):

    f = tables.open_file(filename, mode='r')
    x_array = f.root.data[i*chunk_size:(i+1)*chunk_size,:,:,0:1] # e.g. read from disk only this part of the dataset
    obst_array = f.root.data[i*chunk_size:(i+1)*chunk_size,:,:,1:2]
    y_array = f.root.data[i*chunk_size:(i+1)*chunk_size,:,:,2:3]
    f.close()

    x_array_flat = x_array.reshape((x_array.shape[0], x_array.shape[1]*x_array.shape[2], 1 ))
    x_array_flat1 = x_array_flat[...,0:1]/max_abs_fU
    obst_array_flat = obst_array.reshape((obst_array.shape[0], obst_array.shape[1]*obst_array.shape[2], 1 ))/max_abs_dist
    y_array_flat = y_array.reshape((y_array.shape[0], y_array.shape[1]*y_array.shape[2]))/max_abs_p

    input_flat = np.concatenate((x_array_flat1,obst_array_flat) , axis = -1)
    input_flat = input_flat.reshape((input_flat.shape[0],-1))

    input_dask = dask.array.from_array(input_flat, chunks='auto') 
    y_dask = dask.array.from_array(y_array_flat, chunks='auto') 

    scaler = dask_ml.preprocessing.StandardScaler().fit(input_dask)
    scaler1 = dask_ml.preprocessing.StandardScaler().fit(y_dask)

    inputScaled = scaler.transform(input_dask)
    yScaled = scaler1.transform(y_dask)

    ipca_input.partial_fit(inputScaled)
    ipca_p.partial_fit(yScaled)

    print('fited ' + str(i) + '/' + str(N//chunk_size), flush = True)

  global PC_p, PC_input

  pk.dump(ipca_input, open("ipca_input_more.pkl","wb"))
  pk.dump(ipca_p, open("ipca_p_more.pkl","wb"))

  PC_p = np.argmax(ipca_p.explained_variance_ratio_.cumsum() > 0.95)
  PC_input = np.argmax(ipca_input.explained_variance_ratio_.cumsum() > 0.95)

  print('PC_p ' + str(PC_p))
  print('PC_input ' + str(PC_input))

  print(' Total variance from input represented: ' + str(np.sum(ipca_input.explained_variance_ratio_[:PC_input])))
  print(' Total variance from p represented: ' + str(np.sum(ipca_p.explained_variance_ratio_[:PC_p])))

  for i in range(int(N//chunk_size)):

    f = tables.open_file(filename, mode='r')
    x_array = f.root.data[i*chunk_size:(i+1)*chunk_size,:,:,0:1] # e.g. read from disk only this part of the dataset
    obst_array = f.root.data[i*chunk_size:(i+1)*chunk_size,:,:,1:2]
    y_array = f.root.data[i*chunk_size:(i+1)*chunk_size,:,:,2:3]
    f.close()

    x_array_flat = x_array.reshape((x_array.shape[0], x_array.shape[1]*x_array.shape[2], 1 ))
    x_array_flat1 = x_array_flat[...,0:1]/max_abs_fU
    obst_array_flat = obst_array.reshape((obst_array.shape[0], obst_array.shape[1]*obst_array.shape[2], 1 ))/max_abs_dist
    y_array_flat = y_array.reshape((y_array.shape[0], y_array.shape[1]*y_array.shape[2]))/max_abs_p

    input_flat = np.concatenate((x_array_flat1,obst_array_flat) , axis = -1)

    input_flat = input_flat.reshape((input_flat.shape[0],-1))

    input_transf = ipca_input.transform(input_flat)
    y_transf = ipca_p.transform(y_array_flat)

    array_image = np.concatenate((np.expand_dims(input_transf, axis=-1) , np.expand_dims(y_transf, axis=-1)), axis = -1)#, y_array]
    print(array_image.shape)

    f = tables.open_file(filename_flat, mode='a')
    f.root.data_flat.append(array_image)
    f.close()

  client.close()

  f = tables.open_file(filename_flat, mode='r')
  input = f.root.data_flat[...,:PC_input,0] 
  output = f.root.data_flat[...,:PC_p,1] 
  f.close()

  print(' Total variance from input represented: ' + str(np.sum(ipca_input.explained_variance_ratio_[:PC_input])))
  print(' Total variance from p represented: ' + str(np.sum(ipca_p.explained_variance_ratio_[:PC_p])))

  #normalizing the PCA data
  global max_abs_p_PCA

  max_abs_input_PCA = np.max(np.abs(input))
  max_abs_p_PCA = np.max(np.abs(output))
  print( max_abs_input_PCA, max_abs_p_PCA)

  np.savetxt('maxs_PCA', [max_abs_input_PCA, max_abs_p_PCA] )

  x = input/max_abs_input_PCA
  y = output/max_abs_p_PCA

  def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

  x, y = unison_shuffled_copies(x, y)
  print('Data shuffled \n')

  x = x.reshape((x.shape[0], x.shape[1], 1, 1))
  y = y.reshape((y.shape[0], y.shape[1], 1, 1))

  #tf records

  # Convert values to compatible tf.Example types.

  def _bytes_feature(value):
      """Returns a bytes_list from a string / byte."""
      if isinstance(value, type(tf.constant(0))):
          value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
      return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
  def _float_feature(value):
      """Returns a float_list from a float / double."""
      return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))
  def _int64_feature(value):
      """Returns an int64_list from a bool / enum / int / uint."""
      return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

  def parse_single_image(input_parse, output_parse):

    #define the dictionary -- the structure -- of our single example
    data = {
    'height' : _int64_feature(input_parse.shape[0]),
          'depth_x' : _int64_feature(input_parse.shape[1]),
          'depth_y' : _int64_feature(output_parse.shape[1]),
          'raw_input' : _bytes_feature(tf.io.serialize_tensor(input_parse)),
          'output' : _bytes_feature(tf.io.serialize_tensor(output_parse)),
      }

    #create an Example, wrapping the single features
    out = tf.train.Example(features=tf.train.Features(feature=data))

    return out

  def write_images_to_tfr_short(input, output, filename:str="images"):
    filename= filename+".tfrecords"
    writer = tf.io.TFRecordWriter(filename) #create a writer that'll store our data to disk
    count = 0

    for index in range(len(input)):

      #get the data we want to write
      current_input = input[index].astype('float64')
      current_output = output[index].astype('float64')

      out = parse_single_image(input_parse=current_input, output_parse=current_output)
      writer.write(out.SerializeToString())
      count += 1

    writer.close()
    print(f"Wrote {count} elements to TFRecord")
    return count

  split = 0.9
  count = write_images_to_tfr_short(x[:int(split*x.shape[0]),...], y[:int(split*y.shape[0]),...], filename="train_data_more")
  count = write_images_to_tfr_short(x[int(split*x.shape[0]):,...], y[int(split*y.shape[0]):,...], filename="test_data_more")

  return 0 



def load_data_And_train(lr, Nsamples, block_size, delta, features, batch_size, num_principal_comp, model_name, hdf5_path, beta_1, num_epoch):

  def parse_tfr_element(element):
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
    
    input_out= tf.io.parse_tensor(raw_input, out_type=tf.float64)
    output_out = tf.io.parse_tensor(output, out_type=tf.float64)

    return ( input_out , output_out)

  def load_dataset(filename, batch_size, buffer_size):
    #create the dataset
    dataset = tf.data.TFRecordDataset(filename)

      #pass every single feature through our mapping function
    dataset = dataset.map(
        parse_tfr_element
    )

    dataset = dataset.shuffle(buffer_size=buffer_size )
    #epoch = tf.data.Dataset.range(epoch_num)
    dataset = dataset.batch(batch_size)

    return dataset  
    #return tf.compat.v1.data.make_one_shot_iterator(dataset)

  train_path = 'train_data_more.tfrecords'
  test_path = 'test_data_more.tfrecords'

  train_dataset = load_dataset(filename = train_path, batch_size= batch_size, buffer_size=1024)
  test_dataset = load_dataset(filename = test_path, batch_size= batch_size, buffer_size=1024)



  #coding the training


  @tf.function
  def train_step(inputs, labels):
    with tf.GradientTape() as tape:
      predictions = model(inputs, training=True)
      loss=loss_object(labels, predictions) 

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

  #@tf.function
  def perform_validation():

    losses = []

    for (x_val, y_val) in test_dataset:
      x_val = tf.cast(x_val[...,0,0], dtype='float32')
      y_val = tf.cast(y_val[...,0,0], dtype='float32')

      val_logits = model(x_val)
      val_loss = loss_object(y_true = y_val , y_pred = val_logits)
      losses.append(val_loss)

    return losses

  
  def my_mse_loss():
    def loss_f(y_true, y_pred):

      loss = tf.reduce_mean(tf.square(y_true - y_pred) ) 

      return   1e6 * loss
    return loss_f

  #training 

  lr = lr
  features = features

  optimizer = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=beta_1, beta_2=0.999, epsilon=1e-08)#, decay=0.45*lr, amsgrad=True)
  loss_object = my_mse_loss()

  model = DENSE_PCA()

  def Callback_EarlyStopping(LossList, min_delta=0.1, patience=20):
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

  epochs = num_epoch
  epochs_val_losses, epochs_train_losses = [], []

  min_yet = 10000

  optimizer = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=beta_1, beta_2=0.999)#, epsilon=1e-08, decay=0.45*lr, amsgrad=True)

  for epoch in range(epochs):
    progbar = tf.keras.utils.Progbar(math.ceil(len_train/batch_size))
    print('Start of epoch %d' %(epoch,))
    losses_train = []
    losses_test = []

    for step, (inputs, labels) in enumerate(train_dataset):

      inputs = tf.cast(inputs[...,0,0], dtype='float32')
      labels = tf.cast(labels[...,0,0], dtype='float32')
      loss = train_step(inputs, labels)
      losses_train.append(loss)

    losses_val  = perform_validation()

    losses_train_mean = np.mean(losses_train)
    losses_val_mean = np.mean(losses_val)

    epochs_train_losses.append(losses_train_mean)
    epochs_val_losses.append(losses_val_mean)
    print('Epoch %s: Train loss: %.4f , Validation Loss: %.4f \n' % (epoch,float(losses_train_mean), float(losses_val_mean)))

    progbar.update(step+1)

    stopEarly = Callback_EarlyStopping(epochs_val_losses, min_delta=0.1/100, patience=250)
    if stopEarly:
      print("Callback_EarlyStopping signal received at epoch= %d/%d"%(epoch,epochs))
      print("Terminating training ")
      mod = 'model_' + model_name + '_.h5'
      break
    
    if epoch > 20 and losses_val_mean < min_yet:
      min_yet = losses_val_mean
      print('saving model')
      mod = 'model_' + model_name + '_.h5'
      model.save(mod)
    
  np.savetxt('train_loss' + str(beta_1)+ str(lr)+ '.txt', epochs_train_losses, fmt='%d')
  np.savetxt('test_loss' + str(beta_1)+ str(lr)+ '.txt', epochs_val_losses, fmt='%d')
      
  return 0


hdf5_path = 'dataset_unsteadyCil_fu_bound.hdf5' #adjust path


# DATA-Processing PARAMETERS

Nsamples = 5e5/10
block_size = 128
delta = 5e-3
modes_PCA = 512
input_shape = (np.sqrt(modes_PCA), np.sqrt(modes_PCA), 3)

import pickle as pk

ipca_p = pk.load(open("ipca_p_more.pkl",'rb'))
ipca_input = pk.load(open("ipca_input_more.pkl",'rb'))

#prepare_data (block_size, Nsamples, modes_PCA) #prepare and save data to tf records
len_train = int(0.9*Nsamples*num_sims)

PC_p = 39
PC_input = 116

PC_input = int(PC_input)
PC_p = int(PC_p)

def DENSE_PCA(input_shape = (PC_input)):

    inputs = Input(input_shape)

    x = tf.keras.layers.Dense(512, activation='relu')(inputs)
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    # x = tf.keras.layers.Dense(512)(x)
    # x = tf.keras.layers.Dense(512)(x)
    outputs = tf.keras.layers.Dense(PC_p)(x) #shape(16,16,1)

    model = Model(inputs, outputs, name="U-Net")
    print(model.summary())

    return model

# #if not prepare data
ipca_p = pk.load(open("ipca_p_more.pkl",'rb'))
maxs_PCA = np.loadtxt('maxs_PCA')
max_abs_input_PCA, max_abs_p_PCA = maxs_PCA[0], maxs_PCA[1]

# TRAINing PARAMETERS

num_epoch = 5000

lr = 1e-4#[5e-4, 1e-4, 5e-5]#, 1e-5]
beta = 0.99#, 0.9]


filters = 32*2
batch_size = 1024#*8
model_name = 'first'


load_data_And_train(lr, Nsamples, block_size, delta, filters, batch_size, modes_PCA ,model_name,hdf5_path,beta_1, num_epoch )

