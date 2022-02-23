import numpy as np
import h5py
import os
from tqdm import tqdm
import pyvista as pv

def reduce(list1,factor):
      list2 = []
      list3 = []
      print(list1[0].shape)
      for i in range(int(list1[0].shape[0]/factor)):
         list2.append(list1[0][i*factor])
      print(len(list2))
      arr = np.array(list2)
      list3.append(arr)
      return list3

def reduce_points(array,factor):
   new_array = []
   for i in range(int(array.shape[0]/factor)):
      new_array.append(array[i*factor])

   new_array = np.array(new_array)
   return new_array

def padding(array):
        x=[]
        max = 12000
        a =np.pad(array, [(0, max - array.shape[0] ) ], mode = 'constant', constant_values = -100.0)
        x.append(a)
        x= np.array(x).reshape((max,))
        return x

def padding_b(array):
        x=[]
        max = 1000
        a =np.pad(array, [(0, max - array.shape[0] ) ], mode = 'constant', constant_values = -100.0)
        x.append(a)
        x= np.array(x).reshape((max,))
        return x

def extract_simulation_data(serial=None):
    data = {'Ux': [], 'Uy': [], 'p': [], 'Cx': [], 'Cy': [] }
    for entity in ['p', 'U', 'Cx', 'Cy']:
      for time in range(int(total_times)):
	          
        vtk_list = []
        path = 'simulation_data/' + serial + '/VTK/' + serial + '_' + str(int(time * 100 * deltat_write)) + '.vtk'
        mesh_cell_data = pv.read(path).cell_arrays 
        vtk_list.append(reduce_points(mesh_cell_data[entity],1))
        #vtk_list = reduce(vtk_list,10)        

        if entity == 'U':
          data['Ux'].append(padding(np.concatenate(vtk_list)[:,0]))
          data['Uy'].append(padding(np.concatenate(vtk_list)[:,1]))

        else:
          extent = 0, 3, 0, 1
          data[entity].append(padding(np.concatenate(vtk_list)))

    return data


def extract_simulation_data_boundaries(serial=None):
    data = {'Ux': [], 'Uy': [], 'p': [], 'Cx': [], 'Cy': [] }
    for entity in ['U', 'p', 'Cx', 'Cy']:
      for time in range(int(total_times)):
	          
        vtk_list = []
        #boundaries
        for patch in ['inlet', 'obstacle', 'outlet', 'top']:
    
          path = 'simulation_data/' + serial + '/VTK/'+ patch + '/'+ patch + '_' + str(int(time * 100 * deltat_write)) + '.vtk'
          mesh_cell_data = pv.read(path).cell_arrays 
          vtk_list.append(reduce_points(mesh_cell_data[entity],1))

        if entity == 'U':
          data['Ux'].append(padding_b(np.concatenate(vtk_list)[:,0]))
          data['Uy'].append(padding_b(np.concatenate(vtk_list)[:,1]))

        else:
          extent = 0, 3, 0, 1
          data[entity].append(padding_b(np.concatenate(vtk_list)))

    return data



directory_list = next(os.walk('simulation_data/'))[1]
total_sim = len(directory_list)

directory_list_sim = next(os.walk('simulation_data/0/'))[1]
total_times = len(directory_list_sim) - 3 #constant, dynamicCode, system and VTK files - no dyna,iccode
deltat_write = 1

train_shape = (total_sim, int(total_times), 12000 , 5)

hdf5_path = 'dl_data/all_data_vanKarman_reduced.hdf5'
hdf5_file = h5py.File(hdf5_path, mode='w')
hdf5_file.create_dataset('sim_data', train_shape, np.float32)

boundaries_shape = (total_sim, int(total_times), 1000 , 5)

hdf5_file.create_dataset('boundaries_data', boundaries_shape, np.float32)


count = 0

for sim_no in tqdm(directory_list):

    data = extract_simulation_data(sim_no)
    hdf5_file['sim_data'][count, ..., 0] = data['Ux']
    hdf5_file['sim_data'][count, ..., 1] = data['Uy']
    hdf5_file['sim_data'][count, ..., 2] = data['p']
    hdf5_file['sim_data'][count, ..., 3] = data['Cx']
    hdf5_file['sim_data'][count, ..., 4] = data['Cy']

    boundaries_data = extract_simulation_data_boundaries(sim_no)
    hdf5_file['boundaries_data'][count, ..., 0] = boundaries_data['Ux']
    hdf5_file['boundaries_data'][count, ..., 1] = boundaries_data['Uy']
    hdf5_file['boundaries_data'][count, ..., 2] = boundaries_data['p']
    hdf5_file['boundaries_data'][count, ..., 3] = boundaries_data['Cx']
    hdf5_file['boundaries_data'][count, ..., 4] = boundaries_data['Cy']

    count = count + 1

hdf5_file.close()
