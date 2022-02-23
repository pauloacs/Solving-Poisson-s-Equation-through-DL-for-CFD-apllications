import matplotlib.pyplot as plt
from ctypes import py_object
from re import X
from numpy.core.defchararray import array
import tensorflow as tf
from tensorflow.python.ops.gen_array_ops import inplace_update
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
from pyDOE import lhs


def read_dataset(path=None, split=0.9, print_shape=False):

    x = []
    hdf5_file = h5py.File(path, "r")
    x = hdf5_file["sim_data"][:, :, ...]
    hdf5_file.close()

    return x

hdf5_path = 'all_data_vanKarman_reduced.hdf5'
x_test = read_dataset(hdf5_path, split=0.9, print_shape=True)

path = 'coords.hdf5'
hdf5_file = h5py.File(path, "r")

coords = hdf5_file["obstacle"][...]

hdf5_file.close()

@njit
def index(array, item):
    for idx, val in np.ndenumerate(array):
        if val == item:
            return idx

indice = index(np.array(x_test[0,0,:,0]) , -100.0 )[0]

print(x_test.shape)
coords_data = x_test[0:1,0,:indice,3:5]

x_data = coords_data.reshape(1, 1,indice, 2)
y_data = x_test[0:1,1,:indice,0:3].reshape(1, 1,indice, 3)

data = tf.data.Dataset.from_tensor_slices(x_data)

print('x_data.shape'+ str(x_data.shape))


import random

def separate_data(coords):
    x = x_data[0,0:1,:,...]

    indice_cords = index(coords[0,:,0] , -100.0 )[0]

    coords = coords[0:1,:indice_cords,:]

    mask_obstacle = np.zeros((x_data[0,0:1,:,1].shape), dtype = bool)

    for i in range(coords.shape[1]):
      mask_obstacle = mask_obstacle | ( np.isclose(x_data[0,:,:,0:2],  coords[:,i,:])[...,0] & np.isclose(x_data[0,:,:,0:2],  coords[:,i,:])[...,1])

    mask_obstacle = mask_obstacle[:,:]  #excluding the IC

    y_max = np.max(x[:,:,1])
    y_min = np.min(x[:,:,1])

    mask_wall = np.isclose(x[:,:,1],y_max) | np.isclose(x[:,:,1],y_min)
    n_wall = np.sum(mask_wall)
    
    x_max = np.max(x[:,:,0])
    x_min = np.min(x[:,:,0])

    mask_outlet = np.isclose(x[:,:,0],x_max)

    n_outlet = np.sum(mask_wall)

    mask_inlet = np.isclose(x[:,:,0],x_min)

    n_inlet = np.sum(mask_inlet)

    n_obstacle = np.sum(mask_obstacle)

 
    X_inlet = x_data[0,:,:,:][mask_inlet != 0]
    X_outlet = x_data[0,:,:,:][mask_outlet != 0]
    X_wall = x_data[0,:,:,:][mask_wall != 0]
    X_obstacle = x_data[0,:,:,:][mask_obstacle != 0]

    lb = np.array([x_min , y_min])
    ub = np.array([ x_max, y_max ])

    factor = 0.5

    X_in = lb + (ub-lb)*lhs(2,int(factor * 40000))

    def DelCylPT(XY_c, xc=0.0, yc=0.0, r=0.1):
        '''
        delete points within cylinder
        '''
        dst = np.array([((xy[0] - xc) ** 2 + (xy[1] - yc) ** 2) ** 0.5 for xy in XY_c])
        return XY_c[dst>r,:]

    r = np.max(X_obstacle[:,0])/2 - np.min(X_obstacle[:,0])/2
    xc = np.min(X_obstacle[:,0]) + r
    yc = np.min(X_obstacle[:,1]) + r

    #REFINE NEAR THE CYLINDER

    lb_ref_cil = np.array([xc - r*2 , yc - r*1.5])
    ub_ref_cil = np.array([xc + r*2, yc + r*1.5])
    X_in_ref_cil =  lb_ref_cil + (ub_ref_cil-lb_ref_cil)*lhs(2,int(factor * 10000))

    lb_ref_top = np.array([x_min , y_max - 0.05])
    ub_ref_top = np.array([x_max, y_max])
    X_in_ref_top =  lb_ref_top + (ub_ref_top-lb_ref_top)*lhs(2,int(factor * 3000))

    lb_ref_bot = np.array([x_min , y_min ])
    ub_ref_bot = np.array([x_max, y_min + 0.05])
    X_in_ref_bot =  lb_ref_bot + (ub_ref_bot-lb_ref_bot)*lhs(2,int(factor * 3000))
  
    X_in = np.concatenate((X_in, X_in_ref_cil, X_in_ref_top, X_in_ref_bot), axis=0)

    X_in = DelCylPT(X_in, xc , yc , r )

    print(X_in.shape)

    lb_wallup = np.array([x_min,y_max] )
    ub_wallup = np.array([x_max,y_max])
    wall_up = lb_wallup + (ub_wallup - lb_wallup) * lhs(2,int(factor * 500))

    lb_walldown = np.array([x_min,y_min] )
    ub_walldown = np.array([x_max,y_min])
    wall_down = lb_walldown + (ub_walldown - lb_walldown) * lhs(2,int(factor * 500))

    #idx_f_model = random.sample(range(0, X_obstacle.shape[1]), int(0.25 * X_obstacle.shape[1]))
    #X_obstacle = np.take(X_obstacle, idx_f_model ,axis=0)

    X_walls = np.concatenate((wall_up, wall_down ,X_obstacle), axis=0)
    
    lb_inlet = np.array([x_min,y_min])
    ub_inlet = np.array([x_min,y_max])
    X_inlet = lb_inlet + (ub_inlet-lb_inlet)*lhs(2,int(factor * 200))  

    lb_outlet = np.array([x_max,y_min])
    ub_outlet = np.array([x_max,y_max])
    X_outlet = lb_outlet + (ub_outlet-lb_outlet)*lhs(2,int(factor * 200))  


    # n_Samples = int(0.25 * x_data.shape[2])
    # idx_f_model = random.sample(range(0, x_data.shape[2]), n_Samples)
    # X_in = np.take(x_data[0,0,...], idx_f_model ,axis=0)
    # #Y_in = np.take(y_data[0,...], idx_f_model ,axis=0)

    # idx_f_model = random.sample(range(0, X_inlet.shape[1]), int(0.5 * X_inlet.shape[1]))
    # X_inlet = np.take(X_inlet, idx_f_model ,axis=0)

    # idx_f_model = random.sample(range(0, X_outlet.shape[1]), int(0.5 * X_outlet.shape[1]))
    # X_outlet = np.take(X_outlet, idx_f_model ,axis=0)

    # idx_f_model = random.sample(range(0, X_walls.shape[1]), int(0.5 * X_walls.shape[1]))
    # X_walls = np.take(X_walls, idx_f_model ,axis=0)

    X_in = tf.cast(X_in, dtype='float32')
    X_inlet = tf.cast(X_inlet, dtype='float32')
    X_outlet = tf.cast(X_outlet, dtype='float32')
    X_walls = tf.cast(X_walls, dtype='float32')

    return X_inlet, X_outlet, X_walls, X_in


X_inlet, X_outlet, X_walls, X_in = separate_data(coords)

lb = np.min(x_data[0,0:1,:,0:2], axis=1)
ub = np.max(x_data[0,0:1,:,0:2], axis=1)


def normalize_X(X):
  x_norm  = 2.0 *(X[...,0:1]-lb[0][0])/(ub[0][0]-lb[0][0])-1.0 
  y_norm  = 2.0 *(X[...,1:2]-lb[0][1])/(ub[0][1]-lb[0][1])-1.0 
  X_normalized = tf.concat([x_norm, y_norm], axis = -1)
  return X_normalized

def dense_model():
  input_layer = tf.keras.Input (shape=( 2))
  x = tf.keras.layers.Lambda(normalize_X)(input_layer)
  x = tf.keras.layers.Dense(50, activation='tanh')(x)
  x = tf.keras.layers.Dense(50, activation='tanh')(x)
  x = tf.keras.layers.Dense(50, activation='tanh')(x)
  x = tf.keras.layers.Dense(50, activation='tanh')(x)
  x = tf.keras.layers.Dense(50, activation='tanh')(x)
  x = tf.keras.layers.Dense(50, activation='tanh')(x)
  x = tf.keras.layers.Dense(50, activation='tanh')(x)
  output_layer = tf.keras.layers.Dense(3)(x)
  
  model = tf.keras.Model(inputs=[input_layer], outputs = [output_layer])

  print(model.summary())
  return model

@tf.function
def net_uv(data):

    dtype = "float32"
    x = tf.convert_to_tensor(data[...,0:1], dtype=dtype)
    y = tf.convert_to_tensor(data[...,1:2], dtype=dtype)

    u_p_sigma = model_ref(tf.stack([x, y], axis=1))

    u = u_p_sigma[...,0:1]
    v = u_p_sigma[...,1:2]
    p = u_p_sigma[...,2:3]

    return u, v, p

@tf.function
def net_f(data):

    dtype = "float32"
    x = tf.convert_to_tensor(data[...,0:1], dtype=dtype)
    y = tf.convert_to_tensor(data[...,1:2], dtype=dtype)

    with tf.GradientTape(persistent=True) as tape:
      tape.watch(x)
      tape.watch(y)

      u_p_sigma = model_ref(tf.stack([x,y], axis=1))

      u = u_p_sigma[:,0:1]
      v = u_p_sigma[:,1:2]
      p = u_p_sigma[:,2:3]

      u_x = tape.gradient(u,x)
      u_y = tape.gradient(u,y)

      v_x = tape.gradient(v, x)
      v_y = tape.gradient(v,y)

    u_xx = tape.gradient(u_x,x)
    u_yy = tape.gradient(u_y,y)

    v_xx = tape.gradient(v_x,x)
    v_yy = tape.gradient(v_y,y)

    p_x = tape.gradient(p,x)
    p_y = tape.gradient(p,y)

    nu= 0.02
    NSx_residual =  u * u_x + v * u_y + p_x - nu * (u_xx + u_yy)
    NSy_residual =  u * v_x + v * v_y + p_y - nu * (v_xx + v_yy)

    # cont_residual = tf.reduce_mean(tf.square(cont_residual))
    NSx_residual = tf.reduce_mean(tf.square(NSx_residual))
    NSy_residual = tf.reduce_mean(tf.square(NSy_residual))

    loss_equations = NSx_residual + NSy_residual

    return loss_equations

U_mean = 1
h = 0.5

u_inlet_true = 1.5 * U_mean * ( np.ones(X_inlet[:,1:2].shape) - np.square(X_inlet[:,1:2]/h))

#@tf.function
def eq_loss_NS_2():
  def equation_loss(X_inlet, X_outlet, X_walls, X_in, u_inlet_true):

      loss_eq = net_f(X_in)
      u_inlet, v_inlet, _  =  net_uv(X_inlet)
      _ , _ , p_outlet =  net_uv(X_outlet)
      u_walls, v_walls, _ =  net_uv(X_walls)

      loss_wall = tf.reduce_mean(tf.square(u_walls)) + tf.reduce_mean(tf.square(v_walls))
      loss_inlet = tf.reduce_mean(tf.square(u_inlet - u_inlet_true)) + tf.reduce_mean(tf.square(v_inlet))
      loss_outlet = tf.reduce_mean(tf.square(p_outlet))

      return     1000* (loss_eq + 5 *( loss_wall + loss_inlet + loss_outlet)) 

  return equation_loss


# loss_object1 = tf.keras.losses.MeanSquaredError()

# def perform_validation():
#   losses = []

#   for step, (x_batch_test) in enumerate (data):

#     x_batch_test = tf.cast(x_batch_test, dtype='float32')
#     ux,uy,p  = net_uv(x_batch_test[0,...])
#     val_logits = tf.concat([ux, uy, p] , axis=1)


#     #val_loss = loss_object(y_true= y_val , y_pred = val_logits, x=x_val)
#     val_loss = loss_object1(y_true= y_data[0:1,:,0:3] , y_pred = val_logits)
#     losses.append(val_loss)
#     #val_acc_metric(y_val,val_logits)
#   return losses

#coding the training
@tf.function
def apply_gradient(optimizer, model , x):
  with tf.GradientTape() as tape:
    #logits = model_ref(x)  #predictions from the model at this moment
    #loss_value = loss_object(y_true = y , y_pred = logits, x=x) #calculate loss given the predictions - logits
    loss_value = loss_object(X_inlet, X_outlet, X_walls, X_in, u_inlet_true)
  gradients = tape.gradient(loss_value, model.trainable_weights)
  optimizer.apply_gradients(zip(gradients,model.trainable_weights)) #zip to build tuple

  return loss_value # logits, loss_value 

#@tf.function
def train_data_for_one_epoch():
  losses = []  #loss per batch
  
  #pbar=tqdm(total=len(list(enumerate(train))), position=0, leave = True, bar_format = '{1_bar}{bar}| {n_fms}/{total_fms} ')

  for step, (x_batch_train) in enumerate (data):

    x_batch_train = tf.cast(x_batch_train, dtype='float32')
    
    loss_value = apply_gradient(optimizer , model_ref , x_batch_train)

    losses.append(loss_value)

    #train_acc_metric(y_batch_train, logits)
    #pbar.set_description("Training los for step %.4f" % (int(step),float(loss_value)))
    #pbar.update()
    #progbar.update(step+1)

  return losses

model_ref = dense_model()
loss_object = eq_loss_NS_2()
lr = 5e-4
optimizer = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.45*lr, amsgrad=True)


#from tqdm import tqdm 
import math

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

#from tqdm import tqdm 
import math


epochs = 10000
epochs_val_losses, epochs_train_losses = [], []

train_len = 1

for epoch in range(epochs):
  
  #progbar = tf.keras.utils.Progbar(math.ceil(train_len/batch_size))

  print('Start of epoch %d' %(epoch,))

  losses_train = train_data_for_one_epoch()
  #train_acc = train_acc_metric.result()

  # losses_val  = perform_validation()
  # #val_acc = val_acc_metric.result()

  losses_train_mean = np.mean(losses_train)
  #losses_val_mean = np.mean(losses_val)
  #epochs_val_losses.append(losses_val_mean)
  epochs_train_losses .append(losses_train_mean)
  print('Epoch %s: loss: %.4f \n' % (epoch,float(losses_train_mean)), flush = True)
  #print('Epoch %s: loss: %.4f real_loss: %.4f \n' % (epoch,float(losses_train_mean), float(losses_val_mean)), flush = True)
  #train_acc_metric.reset_states()
  #val_acc_metric.reset_states()

  stopEarly = Callback_EarlyStopping(epochs_train_losses, min_delta=0.1/100, patience=500)
  if stopEarly:
    print("Callback_EarlyStopping signal received at epoch= %d/%d"%(epoch,epochs))
    print("Terminating training ")
    break

#model_ref.compile(optimizer,loss_object)
#loss_object = tf.keras.losses.MeanSquaredError()

# score = model_ref.evaluate(x_data[0,...],y_data, verbose=1)

# print('Average Mean Squared Error:', score)
model_ref.save('my_model_ref.h5')

#from google.colab import files
#files.download('my_model_ref.h5') 

#model_ref = tf.keras.models.load_model('my_model_ref.h5')

import numpy
import tensorflow as tf
import tensorflow_probability as tfp


def function_factory(model, loss, x):#, y_prev):
    """A factory to create a function required by tfp.optimizer.lbfgs_minimize.
    Args:
        model [in]: an instance of `tf.keras.Model` or its subclasses.
        loss [in]: a function with signature loss_value = loss(pred_y, true_y).
        train_x [in]: the input part of training data.
        train_y [in]: the output part of training data.
    Returns:
        A function that has a signature of:
            loss_value, gradients = f(model_parameters).
    """

    # obtain the shapes of all trainable parameters in the model
    shapes = tf.shape_n(model.trainable_variables)
    n_tensors = len(shapes)

    # we'll use tf.dynamic_stitch and tf.dynamic_partition later, so we need to
    # prepare required information first
    count = 0
    idx = [] # stitch indices
    part = [] # partition indices

    for i, shape in enumerate(shapes):
        n = numpy.product(shape)
        idx.append(tf.reshape(tf.range(count, count+n, dtype=tf.int32), shape))
        part.extend([i]*n)
        count += n

    part = tf.constant(part)

    @tf.function
    def assign_new_model_parameters(params_1d):
        """A function updating the model's parameters with a 1D tf.Tensor.
        Args:
            params_1d [in]: a 1D tf.Tensor representing the model's trainable parameters.
        """

        params = tf.dynamic_partition(params_1d, part, n_tensors)
        for i, (shape, param) in enumerate(zip(shapes, params)):
            model.trainable_variables[i].assign(tf.reshape(param, shape))

    # now create a function that will be returned by this factory
    @tf.function
    def f(params_1d):
        """A function that can be used by tfp.optimizer.lbfgs_minimize.
        This function is created by function_factory.
        Args:
           params_1d [in]: a 1D tf.Tensor.
        Returns:
            A scalar loss and the gradients w.r.t. the `params_1d`.
        """

        # use GradientTape so that we can calculate the gradient of loss w.r.t. parameters
        with tf.GradientTape() as tape:
            # update the parameters in the model
            assign_new_model_parameters(params_1d)
            # calculate the loss
            loss_value = loss(X_inlet, X_outlet, X_walls, X_in, u_inlet_true)
        # calculate gradients and convert to 1D tf.Tensor
        grads = tape.gradient(loss_value, model.trainable_variables)
        grads = tf.dynamic_stitch(idx, grads)

        #grad_flat = []
        #for g in grads:
        #    grad_flat.append(tf.reshape(g, [-1]))
        #grad_flat = tf.concat(grad_flat, 0)

        ux,uy,p = net_uv(x[0,...])
        val_logits = tf.concat([ux, uy, p] , axis=1)

        #val_loss = loss_object1(y_true= y_data[0,:,0:3] , y_pred = val_logits[:,:])

        # print out iteration & loss
        f.iter.assign_add(1)
        tf.print("Iter:", f.iter, "loss:", loss_value)#, "loss_val:", val_loss)

        # store loss value so we can retrieve later
        tf.py_function(f.history.append, inp=[loss_value], Tout=[])

        return loss_value, grads

    # store these information as members so we can use them outside the scope
    f.iter = tf.Variable(0)
    f.idx = idx
    f.part = part
    f.shapes = shapes
    f.assign_new_model_parameters = assign_new_model_parameters
    f.history = []

    return f

from matplotlib import pyplot

def plot_helper(inputs, outputs, title, fname):
    """Plot helper"""
    pyplot.figure()
    pyplot.tricontourf(inputs[:, 0], inputs[:, 1], outputs.flatten(), 100)
    pyplot.xlabel("x")
    pyplot.ylabel("y")
    pyplot.title(title)
    pyplot.colorbar()
    pyplot.savefig(fname)


#model = load_model('my_model.h5')
# use float64 by default
#tf.keras.backend.set_floatx("float32")

for step, (x_batch_train) in enumerate (data):
  x = tf.cast(x_batch_train, dtype='float32')

loss = eq_loss_NS_2()
func = function_factory(model_ref, loss, x)#, y_prev)

# convert initial model parameters to a 1D tf.Tensor
init_params = tf.dynamic_stitch(func.idx, model_ref.trainable_variables)

# train the model with L-BFGS solver
results = tfp.optimizer.lbfgs_minimize(value_and_gradients_function=func, initial_position=init_params, max_iterations=100000, tolerance=1e-8)

# after training, the final optimized parameters are still in results.position
# so we have to manually put them back to the model
func.assign_new_model_parameters(results.position)

# print out history
print("\n"+"="*80)
print("History")
print("="*80)
print(*func.history, sep='\n')

model_ref.save('my_model_ref_afterLFGS.h5')

model_ref = tf.keras.models.load_model('my_model_ref_afterLFGS.h5')

def read_dataset(path=None, split=0.9, print_shape=False):

    x = []
    hdf5_file = h5py.File(path, "r")
    x = hdf5_file["sim_data"][:, :, ...]
    hdf5_file.close()

    return x


hdf5_path = 'all_data_vanKarman_reduced.hdf5'
x_test = read_dataset(hdf5_path, split=0.9, print_shape=True)


@njit
def index(array, item):
    for idx, val in np.ndenumerate(array):
        if val == item:
            return idx


indice = index(np.array(x_test[0, 0, :, 0]), -100.0)[0]

print(x_test.shape)
coords_data = x_test[0:1, 0, :indice, 3:5]

x_data = coords_data.reshape(1, 1, indice, 2)

y_data = x_test[0:1, 1, :indice, 0:3].reshape(1, 1, indice, 3)

data = tf.data.Dataset.from_tensor_slices(x_data)

print('x_data.shape' + str(x_data.shape))
print('y_data.shape' + str(y_data.shape))


ux, uy, p = net_uv(x_data[0, 0, ...])
ux_in, uy_in, p_in = net_uv(X_in)

print(ux.shape)
print(y_data.shape)

fig, ax = plt.subplots(nrows=6, figsize=(30, 35))
fig.subplots_adjust(hspace=0.2, wspace=0.02)

u_max = np.max(y_data[..., 0])
u_min = np.min(y_data[..., 0])

v_max = np.max(y_data[..., 1])
v_min = np.min(y_data[..., 1])

p_max = np.max(y_data[..., 2])
p_min = np.min(y_data[..., 2])

ax[0].set_title('PINN prediction - Ux')
cf = ax[0].scatter(X_in[:, 0], X_in[:, 1], c=ux_in[:, 0],
                   edgecolors='none', cmap='jet', marker='.', vmax=u_max, vmin=u_min)
fig.colorbar(cf, ax=ax[0])

# ax[0].set_title('Ground true - CFD - Ux')
# cf = ax[0].scatter(x_data[0,0,:,0], x_data[0,0,:,1], c = ux[...,0], edgecolors='none', cmap='jet', marker='.', vmax=u_max,vmin=u_min )
# fig.colorbar(cf, ax=ax[0])

ax[1].set_title('Ground true - CFD - Ux')
cf = ax[1].scatter(x_data[0, 0, :, 0], x_data[0, 0, :, 1], c=y_data[0, 0, :, 0],
                   edgecolors='none', cmap='jet', marker='.', vmax=u_max, vmin=u_min)
fig.colorbar(cf, ax=ax[1])

ax[2].set_title('PINN prediction - Uy')
cf = ax[2].scatter(x_data[0, 0, :, 0], x_data[0, 0, :, 1], c=uy[:, 0],
                   edgecolors='none', cmap='jet', marker='.', vmax=v_max, vmin=v_min)
fig.colorbar(cf, ax=ax[2])

ax[3].set_title('Ground true - CFD - Uy')
cf = ax[3].scatter(x_data[0, 0, :, 0], x_data[0, 0, :, 1], c=y_data[0, 0, :, 1],
                   edgecolors='none', cmap='jet', marker='.', vmax=v_max, vmin=v_min)
fig.colorbar(cf, ax=ax[3])

ax[4].set_title('PINN prediction - p')
cf = ax[4].scatter(x_data[0, 0, :, 0], x_data[0, 0, :, 1], c=p[:, 0],
                   edgecolors='none', cmap='jet', marker='.', vmax=p_max, vmin=p_min)
fig.colorbar(cf, ax=ax[4])

ax[5].set_title('Ground true - CFD - p')
cf = ax[5].scatter(x_data[0, 0, :, 0], x_data[0, 0, :, 1], c=y_data[0, 0, :, 2],
                   edgecolors='none', cmap='jet', marker='.', vmax=p_max, vmin=p_min)
fig.colorbar(cf, ax=ax[5])

plt.show()
plt.savefig('PINN_sigma.png', dpi=150)
plt.close()

#plot pressure along the cylinder


def separate_data_obst(coords):
    x = x_data[0, 0:1, :, ...]

    indice_cords = index(coords[0, :, 0], -100.0)[0]

    coords = coords[0:1, :indice_cords, :]

    mask_obstacle = np.zeros((x_data[0, 0:1, :, 1].shape), dtype=bool)

    for i in range(coords.shape[1]):
      mask_obstacle = mask_obstacle | (np.isclose(x_data[0, :, :, 0:2],  coords[:, i, :])[
                                       ..., 0] & np.isclose(x_data[0, :, :, 0:2],  coords[:, i, :])[..., 1])

    mask_obstacle = mask_obstacle[:, :]  # excluding the IC

    n_obstacle = np.sum(mask_obstacle)

    X_obstacle = x_data[0, :, :, :][mask_obstacle != 0]

    return X_obstacle


X_obstacle = separate_data_obst(coords)
ux_obst, uy_obst, p_obst = net_uv(X_obstacle)

p_cfd = y_data[..., 2][x_data == X_obstacle]

print(p_obst.shape)

R = (np.max(X_obstacle[:, 0]) - np.min(X_obstacle[:, 0]))/2

center_x = R + np.min(X_obstacle[:, 0])
center_y = R + np.min(X_obstacle[:, 1])

X = X_obstacle[:, 0] - center_x
Y = X_obstacle[:, 1] - center_y

theta = np.arccos((X)/R) * 180/math.pi
theta[Y < 0] = 360-theta[Y < 0]

# theta, p_obst = zip(*sorted(zip(theta, p_obst)))

idx = np.argsort(theta)
theta = theta[idx]
p_obst = np.array(p_obst)[idx]

plt.figure(figsize=(20, 10))


#### getting the cfd values
mask = np.zeros(shape=x_data[0, 0, :, 0].shape)
for x_y in X_obstacle:
  x_y = x_y
  for i in range(x_data.shape[2]):
    if x_data[0, 0, i, 0] == x_y[0] and x_data[0, 0, i, 1] == x_y[1]:
      mask[i] = int(1)

mask = mask.astype(int)

p_cfd_obst = y_data[0, 0, :, 2][mask == 1][idx]
################

plt.plot(theta, p_obst, marker='*', label='Model 3')
plt.plot(theta, p_cfd_obst, marker='x', label='CFD')
plt.legend()
plt.show()
plt.savefig('pressure_plot.png', dpi=150)

