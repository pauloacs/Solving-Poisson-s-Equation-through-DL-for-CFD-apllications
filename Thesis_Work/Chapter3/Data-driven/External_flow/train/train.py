
from keras.callbacks import EarlyStopping, TensorBoard
import tensorflow as tf
import os
import shutil
import time
import h5py
import keras
import numpy as np
import math
from numba import njit


def read_dataset(path=None, split=0.9, print_shape=False):

    x = []
    y = []
    hdf5_file = h5py.File(path, "r")
    data = hdf5_file["sim_data"][:, ...]

    data = data[:,:,:,:]

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

    total_sim = (x.shape[0])/times  

    x_train = x[:(int(total_sim * split)*times), ...]
    x_test = x[(int(total_sim * split)*times):int(total_sim)*times, ...]

    y_train = y[:(int(total_sim * split)*times), ...]
    y_test = y[(int(total_sim * split)*times):int(total_sim)*times, ...]

    if print_shape:
        print("total_sim: {}\nx_train.shape: {}\ny_train.shape: {}\nx_test.shape: {}\ny_test.shape: {}\n".format(
            total_sim,
            x_train.shape,
            y_train.shape,
            x_test.shape,
            y_test.shape))

    return x_train, y_train, x_test, y_test

#hdf5_path = '/homes/up201605045/CONV_PointNet_NS_unsteady/data_NS_unsteady.hdf5'
#x_train, y_train, x_test, y_test = read_dataset(hdf5_path, split=0.9, print_shape=True)

#lb = x_train[...,3:6].min()
#ub = x_train[...,3:6].max()

def normalize_X(X):
  X_normalized  = 2.0 *(X[...,3:6]-lb)/(ub-lb)-1.0 
  X_normalized = tf.concat([X[...,0:3], X_normalized], axis = -1)
  return X_normalized

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

def parse_single_image(image, label):

  #define the dictionary -- the structure -- of our single example
  data = {
	'height' : _int64_feature(image.shape[0]),
        'depth_x' : _int64_feature(image.shape[1]),
        'depth_y' : _int64_feature(label.shape[1]),
        'raw_image' : _bytes_feature(tf.io.serialize_tensor(image)),
        'label' : _bytes_feature(tf.io.serialize_tensor(label)),
    }

  #create an Example, wrapping the single features
  out = tf.train.Example(features=tf.train.Features(feature=data))

  return out


def write_images_to_tfr_short(images, labels, filename:str="images"):
  filename= filename+".tfrecords"
  writer = tf.io.TFRecordWriter(filename) #create a writer that'll store our data to disk
  count = 0

  for index in range(len(images)):

    #get the data we want to write
    current_image = images[index].astype('float64')
    current_label = labels[index].astype('float64')

    out = parse_single_image(image=current_image, label=current_label)
    writer.write(out.SerializeToString())
    count += 1

  writer.close()
  print(f"Wrote {count} elements to TFRecord")
  return count

#count = write_images_to_tfr_short(x_train[:,:,0:5], y_train[:,:,0:3], filename="small_images_train_point_reduced")
#count = write_images_to_tfr_short(x_test[:,:,0:5], y_test[:,:,0:3], filename="small_images_test_point_reduced")


def parse_tfr_element(element):
  #use the same structure as above; it's kinda an outline of the structure we now want to create
  data = {
      'height': tf.io.FixedLenFeature([], tf.int64),
      'label' : tf.io.FixedLenFeature([], tf.string),
      'raw_image' : tf.io.FixedLenFeature([], tf.string),
      'depth_x':tf.io.FixedLenFeature([], tf.int64),
      'depth_y':tf.io.FixedLenFeature([], tf.int64)
    }

  content = tf.io.parse_single_example(element, data)
  
  height = content['height']
  depth_x = content['depth_x']
  depth_y = content['depth_y']
  label = content['label']
  raw_image = content['raw_image']
  
  
  #get our 'feature'-- our image -- and reshape it appropriately
  
  feature= tf.io.parse_tensor(raw_image, out_type=tf.float64)
  feature_1 = feature[:,0:3]
  feature_2 = feature[:,3:5]
  label = tf.io.parse_tensor(label, out_type=tf.float64)


  return ( {"feature_1": feature_1, "feature_2": feature_2} , label)


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
 
train_path = "..."   #set the correct path
test_path = "..."

batch_size = 64

train_dataset = load_dataset(filename = train_path, batch_size= batch_size, buffer_size=1024)
test_dataset = load_dataset(filename = test_path, batch_size= batch_size, buffer_size=1024)

#train_len = x_train.shape[0]
x_train, x_test, y_train,  y_test = None, None, None, None


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

#    x = tf.keras.layers.Lambda(normalize_X)(inputs_coordinates) #uncomment to normalize 

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

from numba import njit

@njit
def index(array, item):
    for idx, val in np.ndenumerate(array):
        if val == item:
            return idx
    # If no item was found return None, other return types might be a problem due to
    # numbas type inference.


def my_mse_loss():
  def loss_f(y_true, y_pred):

    loss = 0

    for i in range(y_true.shape[0]):
      indice = index(np.array(y_true[i,:,0]) , -100.0 )[0]

      ux = y_true[i,:indice,0]
      uy = y_true[i,:indice,1]
      p = y_true[i,:indice,2]

      ux_pred = y_pred[i,:indice,0]
      uy_pred = y_pred[i,:indice,1]
      p_pred = y_pred[i,:indice,2]
     
      loss += tf.reduce_mean(tf.square(ux - ux_pred) + \
                  tf.square(uy - uy_pred) + \
                  tf.square(p - p_pred))

    loss /= y_true.shape[0]  

    return   loss * 1e6

  return loss_f


#training 

print('lr=5e-4- batch128')
lr = 5e-4
optimizer = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.45*lr, amsgrad=True)
loss_object = my_mse_loss()


#coding the training
#@tf.function
def apply_gradient(optimizer, model , x1, x2, y):
  with tf.GradientTape() as tape:
    logits = model([x1,x2])  #predictions from the model at this moment
    #loss_value = loss_object(y_true = y , y_pred = logits, x=x) #calculate loss given the predictions - logits
    loss_value = loss_object(y_true = y , y_pred = logits)
  gradients = tape.gradient(loss_value, model.trainable_weights)
  optimizer.apply_gradients(zip(gradients,model.trainable_weights)) #zip to build tuple

  return logits, loss_value


#@tf.function
def train_data_for_one_epoch():
  losses = []  #loss per batch
  
  #pbar=tqdm(total=len(list(enumerate(train))), position=0, leave = True, bar_format = '{1_bar}{bar}| {n_fms}/{total_fms} ')

  for step, (x_batch_train, y_batch_train) in enumerate (train_dataset):

    x_batch_train_1 = tf.cast(x_batch_train['feature_1'], dtype='float32')
    x_batch_train_2 = tf.cast(x_batch_train['feature_2'], dtype='float32')
    y_batch_train = tf.cast(y_batch_train, dtype='float32')
    
    logits , loss_value = apply_gradient(optimizer , model , x_batch_train_1, x_batch_train_2 , y_batch_train)

    losses.append(loss_value)

    #train_acc_metric(y_batch_train, logits)
    #pbar.set_description("Training los for step %.4f" % (int(step),float(loss_value)))
    #pbar.update()
    progbar.update(step+1)

  return losses

#@tf.function
def perform_validation():
  losses = []

  for x_val, y_val in test_dataset:
    x_val_1 = tf.cast(x_val['feature_1'], dtype='float32')
    x_val_2 = tf.cast(x_val['feature_2'], dtype='float32')
    y_val = tf.cast(y_val, dtype='float32')

    val_logits = model([x_val_1 , x_val_2])
    val_loss = loss_object(y_true= y_val , y_pred = val_logits)
    losses.append(val_loss)
  return losses


import math
print('5e-4batch32')
model = keras_model1()
model.load_weights("weights_model_points_NS_120.h5")

epochs = 500
epochs_val_losses, epochs_train_losses = [], []
train_len = 13500


min_yet = 10000

for epoch in range(epochs):
  
  progbar = tf.keras.utils.Progbar(math.ceil(train_len/batch_size))

  print('Start of epoch %d' %(epoch,))

  losses_train = train_data_for_one_epoch()
  #train_acc = train_acc_metric.result()

  losses_val  = perform_validation()
  #val_acc = val_acc_metric.result()

  losses_train_mean = np.mean(losses_train)
  losses_val_mean = np.mean(losses_val)
  epochs_val_losses.append(losses_val_mean)
  epochs_train_losses .append(losses_train_mean)

  print('Epoch %s: Train loss: %.4f , Validation Loss: %.4f , Train Accuracy: %.4f , Validation Accuracy: %.4f \n' % (epoch,float(losses_train_mean), float(losses_val_mean), float(losses_train_mean), float(losses_val_mean)))
  #train_acc_metric.reset_states()
  #val_acc_metric.reset_states()
  
  if losses_val_mean < min_yet:
      min_yet = losses_val_mean
      print('saving model')
      mod = 'weights_model_points_NS_' + str(epoch) + '.h5'
      model.save_weights(mod)


