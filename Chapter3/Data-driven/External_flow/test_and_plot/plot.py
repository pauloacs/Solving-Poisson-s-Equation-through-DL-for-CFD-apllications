from cmath import nan
from ctypes import py_object
from re import X
import matplotlib
from numpy.core.defchararray import array
import tensorflow as tf
from tensorflow.python.ops.gen_array_ops import inplace_update
physical_devices = tf.config.list_physical_devices('GPU') 
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)



import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from scipy.interpolate import griddata
import numpy as np

from scipy.interpolate import griddata
from scipy.spatial import cKDTree
from scipy.interpolate.interpnd import _ndim_coords_from_arrays

from numba import njit
import numpy as np
import h5py
import matplotlib.pyplot as plt
import math

def read_dataset(path=None, split=0.9, print_shape=False):

    x = []
    y = []
    hdf5_file = h5py.File(path, "r")
    data = hdf5_file["sim_data"][25:28, ...]    

    times = data.shape[1] - 1
    hdf5_file.close()

    for j in range(data.shape[0]):
      for i in range(data.shape[1]-1):
        x.append(data[j, i, :,:])
        y.append(data[j, i+1,:,:])

    data = None

    x = np.array(x)
    y = np.array(y)
    y= y[:,:,:3]


    @njit
    def index(array, item):
      for idx, val in np.ndenumerate(array):
        if val == item:
            return idx
    # If no item was found return None, other return types might be a problem due to
    # numbas type inference.


    max_0 = np.max( x[...,0] )
    max_1 = np.max( x[...,1] )
    max_2 = np.max( x[...,2] )

    x_mod = x * (x!=-100)

    min_0 = np.min( x_mod[...,0] )
    min_1 = np.min( x_mod[...,1] )
    min_2 = np.min( x_mod[...,2] )



    for i in range(int(y.shape[0]/times)):

      indice = index(y[i*times+1,:,0] , -100.0 )[0]
      y[times*i:times*(i+1),:indice,0] = (y[times*i:times*(i+1),:indice,0] - np.ones((y[times*i:times*(i+1),:indice,0].shape))*min_0)/(max_0-min_0)
      y[times*i:times*(i+1),:indice,1] = (y[times*i:times*(i+1),:indice,1] - np.ones((y[times*i:times*(i+1),:indice,1].shape))*min_1)/(max_1-min_1)
      y[times*i:times*(i+1),:indice,2] = (y[times*i:times*(i+1),:indice,2] - np.ones((y[times*i:times*(i+1),:indice,2].shape))*min_2)/(max_2-min_2)


    for i in range(int(x.shape[0]/times)):

      indice = index(x[i*times+1,:,0] , -100.0 )[0]
      x[times*i:times*(i+1),:indice,0] = (x[times*i:times*(i+1),:indice,0] - np.ones((x[times*i:times*(i+1),:indice,0].shape))*min_0)/(max_0-min_0)
      x[times*i:times*(i+1),:indice,1] = (x[times*i:times*(i+1),:indice,1] - np.ones((x[times*i:times*(i+1),:indice,1].shape))*min_1)/(max_1-min_1)
      x[times*i:times*(i+1),:indice,2] = (x[times*i:times*(i+1),:indice,2] - np.ones((x[times*i:times*(i+1),:indice,2].shape))*min_2)/(max_2-min_2)



    t = np.zeros(shape=(x.shape[0],x.shape[1],1))

    for i in range(x.shape[0]):
      t[i,:,:] = np.ones((x.shape[1],1)) * (i - math.floor(i/(times+1))*(times+1)+1)

    x = np.concatenate((x,t), axis=2)

    total_sim = (x.shape[0])/times  #divide this way to make testing easier - x_test[0] will be the first time step of one sim

    x_train = x[:(int(total_sim * split)*times), ...]
    x_test = x[(int(total_sim * split)*times):int(total_sim)*times, ...]
    x = None

    y_train = y[:(int(total_sim * split)*times), ...]
    y_test = y[(int(total_sim * split)*times):int(total_sim)*times, ...]
    y = None

    if print_shape:
        print("total_sim: {}\nx_train.shape: {}\ny_train.shape: {}\nx_test.shape: {}\ny_test.shape: {}\n".format(
            total_sim,
            x_train.shape,
            y_train.shape,
            x_test.shape,
            y_test.shape))

    return x_train, y_train, x_test, y_test


hdf5_path = 'data_NS_unsteady.hdf5'
x_train, y_train, x_test, y_test = read_dataset(hdf5_path, split=0.9, print_shape=True)


lb = x_train[...,3:6].min()
ub = x_train[...,3:6].max()

def normalize_X(X):
  X_normalized  = 2.0 *(X[...,3:6]-lb)/(ub-lb)-1.0 
  X_normalized = tf.concat([X[...,0:3], X_normalized], axis = -1)
  return X_normalized


#model

import tensorflow as tf

def conv_bn(x, filters):
    x = tf.keras.layers.Conv1D(filters, kernel_size=1, padding="valid")(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.0)(x)
    return tf.keras.layers.Activation("relu")(x)


def dense_bn(x, filters, activation = "relu"):
    x = tf.keras.layers.Dense(filters)(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.0)(x)
    return tf.keras.layers.Activation(activation)(x)
    
class OrthogonalRegularizer(tf.keras.regularizers.Regularizer):
    def __init__(self, num_features, l2reg=0.001):
        self.num_features = num_features
        self.l2reg = l2reg
        self.eye = tf.eye(num_features)

    def __call__(self, x):
        x = tf.reshape(x, (-1, self.num_features, self.num_features))
        xxt = tf.tensordot(x, x, axes=(2, 2))
        xxt = tf.reshape(xxt, (-1, self.num_features, self.num_features))
        return tf.reduce_sum(self.l2reg * tf.square(xxt - self.eye))

def tnet(inputs, num_features):

    # Initalise bias as the indentity matrix
    bias = tf.keras.initializers.Constant(np.eye(num_features).flatten())
    reg = OrthogonalRegularizer(num_features)

    x = conv_bn(inputs, 32)
    x = conv_bn(x, 64)
    x = conv_bn(x, 512)
    x = tf.keras.layers.GlobalMaxPooling1D()(x)
    x = dense_bn(x, 256)
    x = dense_bn(x, 128)
    x = tf.keras.layers.Dense(
        num_features * num_features,
        kernel_initializer="zeros",
        bias_initializer=bias,
        activity_regularizer=reg,
    )(x)
    feat_T = tf.keras.layers.Reshape((num_features, num_features))(x)
    # Apply affine transformation to input features
    return tf.keras.layers.Dot(axes=(2, 1))([inputs, feat_T])



def inception_module(filters=None, inputs=None):
    tower_0 = tf.keras.layers.Conv1D(int(filters / 4), 1, padding='same', activation='relu')(inputs)

    tower_1 = tf.keras.layers.Conv1D(int(filters / 4), 1, padding='same', activation='relu')(inputs)
    tower_1 = tf.keras.layers.Conv1D(int((filters * 3) / 8), 3, padding='same', activation='relu')(tower_1)

    tower_2 = tf.keras.layers.Conv1D(int(filters / 8), 1, padding='same', activation='relu')(inputs)
    tower_2 = tf.keras.layers.Conv1D(int(filters / 8), 5, padding='same', activation='relu')(tower_2)

    tower_3 = tf.keras.layers.MaxPooling1D(3, strides=1, padding='same')(inputs)
    tower_3 = tf.keras.layers.Conv1D(int(filters / 4), 1, padding='same', activation='relu')(tower_3)

    concat = tf.keras.layers.concatenate([tower_0, tower_1, tower_2, tower_3], axis=2)

    return concat


def keras_model1(save_model=True, NUM_POINTS= 12000):
    inputs = tf.keras.Input(( NUM_POINTS, 3))  # Input(shape), return a tensor
    # Initial shape: (None, 50, 150, 4)

    x = tf.keras.layers.Conv1D(8, 3, padding='same', activation='relu')(inputs)  # (None, 50, 150, 16)
    x = tf.keras.layers.Conv1D(8, 3, padding='same', activation='relu')(x)  # (None, 50, 150, 16)

    c1 = inception_module(filters=16, inputs=x)  # (None, 50, 150, 32)
    x = inception_module(filters=16, inputs=c1)
    x = inception_module(filters=16, inputs=x)
    x = tf.keras.layers.MaxPooling1D(pool_size= 2)(x)  # (None, 25, 75, 32)

    c2 = inception_module(filters=32, inputs=x)  # (None, 25, 75, 64)
    x = inception_module(filters=32, inputs=c2)
    x = inception_module(filters=32, inputs=x)
    x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)  # (None, 12, 37, 64)

    c3 = inception_module(filters=64, inputs=x)  # (None, 12, 37, 128)
    x = inception_module(filters=64, inputs=c3)
    x = inception_module(filters=64, inputs=x)
    x = tf.keras.layers.MaxPooling1D(pool_size= 2)(x)  # (None, 6, 18, 128)

    c4 = inception_module(filters=128, inputs=x)  # (None, 6, 18, 256)
    x = inception_module(filters=128, inputs=c4)
    x = inception_module(filters=128, inputs=x)
    x = tf.keras.layers.MaxPooling1D(pool_size= 2)(x)  # (None, 3, 9, 256)

    x = inception_module(filters=256, inputs=x)  # (None, 3, 9, 512)
    x = inception_module(filters=256, inputs=x)
    x = inception_module(filters=256, inputs=x)  # (None, 3, 9, 512)

    x = tf.keras.layers.Conv1DTranspose(128, 2, strides=2, padding='same')(x)
    x = tf.keras.layers.concatenate([x, c4], axis=2)  # (None, 3, 9, 512) -> (None, 6, 18, 256)
    x = inception_module(filters=128, inputs=x)
    x = inception_module(filters=128, inputs=x)
    x = inception_module(filters=128, inputs=x)

    x = tf.keras.layers.concatenate(
        [tf.keras.layers.Conv1DTranspose(64, 2, strides= 2, padding='same')(x),
         c3], axis=2)  # (None, 6, 18, 256) -> (None, 12, 36, 128) -> (None, 12, 37, 128)
    x = inception_module(filters=64, inputs=x)
    x = inception_module(filters=64, inputs=x)
    x = inception_module(filters=64, inputs=x)

    x = tf.keras.layers.concatenate(
        [tf.keras.layers.Conv1DTranspose(32, 2, strides=2, padding='same')(x),
         c2], axis=2)  # (None, 12, 37, 128) -> (None, 24, 74, 64) -> (None, 25, 75, 64)
    x = inception_module(filters=32, inputs=x)
    x = inception_module(filters=32, inputs=x)
    x = inception_module(filters=32, inputs=x)

    x = tf.keras.layers.concatenate(
        [tf.keras.layers.Conv1DTranspose(32, 2, strides=2, padding='same')(x),
         c1], axis=2)  # (None, 25, 75, 64) -> (None, 50, 150, 32)

    x = inception_module(filters=16, inputs=x)
    x = inception_module(filters=16, inputs=x)
    x = inception_module(filters=16, inputs=x)

    layer_var = tf.keras.layers.Conv1D(3, 1, activation='sigmoid')(x)


  #spatial info:
    inputs_coordinates = tf.keras.Input(shape=(NUM_POINTS, 2))

#    x = tf.keras.layers.Lambda(normalize_X)(inputs_coordinates)

    x = tnet(inputs_coordinates, 2)
    x = conv_bn(x, 32)
    x = conv_bn(x, 32)
    x = tnet(x, 32)
    x1 = conv_bn(x, 32)
    x = conv_bn(x1, 64)
    x = conv_bn(x, 256)
    global_feature = tf.keras.layers.GlobalMaxPooling1D()(x)
    global_feature_layer = tf.reshape(global_feature,(-1,  1, 256))
    global_feature_layer = tf.tile(global_feature_layer,(1 , NUM_POINTS, 1))
    x = tf.keras.layers.concatenate([x1, global_feature_layer, layer_var ],axis=2)



    x = conv_bn(x, 128)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = conv_bn(x, 64)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = conv_bn(x, 32)
    #x = layers.Dropout(0.3)(x)
    x = dense_bn(x, 64)
    outputs = dense_bn(x, 3, activation="sigmoid")

    
    model = tf.keras.Model(inputs=[inputs, inputs_coordinates], outputs = [outputs])

    model.summary()
    tf.keras.utils.plot_model(model, to_file='model_architecture_with_inception2.png', show_shapes=True, show_layer_names=False)

    return model


model = keras_model1()
model.load_weights("weights.h5")


def points_to_image(xs, ys, values):
    threshold = 0.05
    x = np.arange( np.min(xs), np.max(xs), 0.02)
    y = np.arange( np.min(ys), np.max(ys), 0.02)
    grid = np.meshgrid(x,y)
    points = np.stack((xs,ys),axis=1)
    img = griddata(points, values, tuple(grid), method='linear', fill_value=nan)

    tree = cKDTree(points)
    xi = _ndim_coords_from_arrays(tuple(grid), ndim=points.shape[1])
    dists, indexes = tree.query(xi[int(img.shape[0]/2 - img.shape[0]/5): int(img.shape[0]/2 + img.shape[0]/5),int(img.shape[1]/2 - img.shape[1]/4): int(img.shape[1]/2) ]) 
    img[int(img.shape[0]/2 - img.shape[0]/5): int(img.shape[0]/2 + img.shape[0]/5),int(img.shape[1]/2 - img.shape[1]/4): int(img.shape[1]/2) ][dists > threshold] = np.nan
    return img

@njit
def index(array, item):
    for idx, val in np.ndenumerate(array):
        if val == item:
            return idx
    # If no item was found return None, other return types might be a problem due to
    # numbas type inference.


sim= 0


BIAS_u = []
STDE_u = []
RMSE_u = []

BIAS_v = []
STDE_v = []
RMSE_v = []

BIAS_p = []
STDE_p = []
RMSE_p = []


pred_minus_true_u =[]
pred_minus_true_squared_u =[]

pred_minus_true_v =[]
pred_minus_true_squared_v =[]

pred_minus_true_p =[]
pred_minus_true_squared_p =[]

for i in range(12*5):
    i = i*2
    indice = index(x_test[i,:,0], -100)[0]
    x = x_test[i,:indice,3]
    y = x_test[i,:indice,4]

    z_partida = x_test[i,:indice,:]
    z = y_test[i,:indice,:]
    z_model = model([x_test[i,:,0:3].reshape((1,) + (12000,3)), x_test[i,:,3:5].reshape((1,) + (12000,2))])[0,:indice,:]

    u_cfd_ant = points_to_image(x,y,z_partida[:,0])
    v_cfd_ant = points_to_image(x,y,z_partida[:,1])
    p_cfd_ant = points_to_image(x,y,z_partida[:,2])

    u_model = points_to_image(x,y,z_model[:,0])
    v_model = points_to_image(x,y,z_model[:,1])
    p_model = points_to_image(x,y,z_model[:,2])

    u_cfd = points_to_image(x,y,z[:,0])
    v_cfd = points_to_image(x,y,z[:,1])
    p_cfd = points_to_image(x,y,z[:,2])

    mask = [~np.isnan(u_cfd)]
    norm_u = (np.max(u_cfd[mask]) - np.min(u_cfd[mask]))
    norm_v = (np.max(v_cfd[mask]) - np.min(v_cfd[mask]))
    norm_p = (np.max(p_cfd[mask]) - np.min(p_cfd[mask]))

    error_u = np.abs(u_model - u_cfd)/norm_u *100
    error_v = np.abs(v_model - v_cfd)/norm_v *100
    error_p = np.abs(p_model - p_cfd)/norm_p *100

     fig, axs = plt.subplots(3, 3)

     #axs[0,0].set_title('DL prediction', fontsize=25)
     axs[0,0].imshow(u_model, cmap='jet', vmax = 1, vmin=0)
     #axs[0,0].colorbar()
     #axs[0,0].axis('off')
     axs[0,0].set_xticks([])
     axs[0,0].set_yticks([])

     #axs[1,0].set_title('OpenFOAM', fontsize=15)
     axs[1,0].imshow(v_model, cmap='jet', vmax = 1, vmin=0)
     #axs[1,0].colorbar()
     #axs[1,0].axis('off')
     axs[1,0].set_xticks([])
     axs[1,0].set_yticks([])

     #axs[2,0].set_title('partida', fontsize=15)
     axs[2,0].imshow(p_model , cmap='jet', vmax = 1, vmin=0)
     #axs[2,0].colorbar()
     #axs[2,0].axis('off')
     axs[2,0].set_xticks([])
     axs[2,0].set_yticks([])

     axs[0,1].set_title('CFD result', fontsize=25)
     axs[0,1].imshow(u_cfd, cmap='jet', vmax = 1, vmin=0)
     #axs[0,1].colorbar()
     #axs[0,1].axis('off')
     axs[0,1].set_xticks([])
     axs[0,1].set_yticks([])

     #axs[1,1].set_title('OpenFOAM', fontsize=15)
     axs[1,1].imshow(v_cfd, cmap='jet', vmax = 1, vmin=0)
     #axs[1,1].colorbar()
     #axs[1,1].axis('off')
     axs[1,1].set_xticks([])
     axs[1,1].set_yticks([])

     #axs[2,1].set_title('partida', fontsize=15)
     axs[2,1].imshow(p_cfd , cmap='jet', vmax = 1, vmin=0)
     #axs[2,1].colorbar()
     #axs[2,1].axis('off')
     axs[2,1].set_xticks([])
     axs[2,1].set_yticks([])

     axs[0,2].set_title('Error (%)', fontsize=15)
     cf = axs[0,2].imshow(error_u, cmap='jet', vmax = 5, vmin=0)
     plt.colorbar(cf, ax=axs[0,2])
     #axs[0,2].axis('off')
     axs[0,2].set_xticks([])
     axs[0,2].set_yticks([])

     #axs[1,1].set_title('OpenFOAM', fontsize=15)
     cf = axs[1,2].imshow(error_v, cmap='jet', vmax = 5, vmin=0)
     plt.colorbar(cf, ax=axs[1,2])
     #axs[1,2].axis('off')
     axs[1,2].set_xticks([])
     axs[1,2].set_yticks([])

     #axs[2,1].set_title('partida', fontsize=15)
     cf = axs[2,2].imshow(error_p , cmap='jet', vmax = 5, vmin=0)
     plt.colorbar(cf, ax=axs[2,2])
     #axs[2,2].axis('off')
     axs[2,2].set_xticks([])
     axs[2,2].set_yticks([])

     cols = ['DL prediction','Reference (CFD)','Absolute Error (%)']
     rows = ['u' , 'v', 'p']

     for ax, col in zip(axs[0], cols):
         ax.set_title(col, fontsize=10)

     for ax, row in zip(axs[:,0], rows):
         ax.set_ylabel(row, rotation=0, fontsize=10)
         ax.yaxis.labelpad = 15


     fig.tight_layout()
     #plt.show()
     plt.savefig('/home/paulo/Desktop/plots/' + str(i) + '.png', dpi = 400)
     plt.close()
    


    BIAS_norm_u = np.mean( (u_model  - u_cfd )[mask] )/norm_u * 100
    RMSE_norm_u = np.sqrt(np.mean( ( u_model  - u_cfd  )[mask]**2 ))/norm_u * 100
    STDE_norm_u = np.sqrt( (RMSE_norm_u**2 - BIAS_norm_u**2) )

    BIAS_norm_v = np.mean( (v_model  - v_cfd )[mask] )/norm_v * 100
    RMSE_norm_v = np.sqrt(np.mean( ( v_model  - v_cfd  )[mask]**2 ))/norm_v * 100
    STDE_norm_v = np.sqrt( (RMSE_norm_v**2 - BIAS_norm_v**2) )

    BIAS_norm_p = np.mean( (p_model  - p_cfd )[mask] )/norm_p * 100
    RMSE_norm_p = np.sqrt(np.mean( ( p_model  - p_cfd  )[mask]**2 ))/norm_p * 100
    STDE_norm_p = np.sqrt( (RMSE_norm_p**2 - BIAS_norm_p**2) )


    BIAS_u.append(BIAS_norm_u)
    RMSE_u.append(RMSE_norm_u)
    STDE_u.append(STDE_norm_u)

    BIAS_v.append(BIAS_norm_v)
    RMSE_v.append(RMSE_norm_v)
    STDE_v.append(STDE_norm_v)

    BIAS_p.append(BIAS_norm_p)
    RMSE_p.append(RMSE_norm_p)
    STDE_p.append(STDE_norm_p)


    pred_minus_true_u.append( np.mean( (u_model  - u_cfd )[mask] )/norm_u )
    pred_minus_true_squared_u.append( np.mean( (u_model  - u_cfd )[mask]**2 )/norm_u**2 )

    pred_minus_true_v.append( np.mean( (v_model  - v_cfd )[mask] )/norm_v )
    pred_minus_true_squared_v.append( np.mean( (v_model  - v_cfd )[mask]**2 )/norm_v**2 )

    pred_minus_true_p.append( np.mean( (p_model  - p_cfd )[mask] )/norm_p )
    pred_minus_true_squared_p.append( np.mean( (p_model  - p_cfd )[mask]**2 )/norm_p**2 )



BIAS_value_u = np.mean(pred_minus_true_u) * 100
RMSE_value_u = np.sqrt(np.mean(pred_minus_true_squared_u)) * 100

STDE_value_u = np.sqrt( RMSE_value_u**2 - BIAS_value_u**2 )

print('BIAS for the sim: ' + str(BIAS_value_u))
print('RMSE for the sim: ' + str(RMSE_value_u))
print('STDE for the sim: ' + str(STDE_value_u))


BIAS_value_v = np.mean(pred_minus_true_v) * 100
RMSE_value_v = np.sqrt(np.mean(pred_minus_true_squared_v)) * 100

STDE_value_v = np.sqrt( RMSE_value_v**2 - BIAS_value_v**2 )

print('BIAS for the sim: ' + str(BIAS_value_v))
print('RMSE for the sim: ' + str(RMSE_value_v))
print('STDE for the sim: ' + str(STDE_value_v))

BIAS_value_p = np.mean(pred_minus_true_p) * 100
RMSE_value_p = np.sqrt(np.mean(pred_minus_true_squared_p)) * 100

STDE_value_p = np.sqrt( RMSE_value_p**2 - BIAS_value_p**2 )

print('BIAS for the sim: ' + str(BIAS_value_p))
print('RMSE for the sim: ' + str(RMSE_value_p))
print('STDE for the sim: ' + str(STDE_value_p))


