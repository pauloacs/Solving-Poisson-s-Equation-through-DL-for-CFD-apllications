import numpy as np
import h5py
import os
from tqdm import tqdm
import pyvista as pv

def padding(array, max):
	x=[]
	a =np.pad(array, [(0, max - array.shape[0] ) ], mode = 'constant', constant_values = -100.0)
	x.append(a)
	x= np.array(x).reshape((max,))
	return x

def extract_simulation_data_bound(serial, patch):
    data = { 'Cx': [], 'Cy': [] }
    for entity in ['Cx', 'Cy']:
      for time in range(int(total_times)):
        vtk_list = []
        t = round(deltat*(time*3+1)*100)/100
        if t % 1 == 0 :
          t = round(t)

	#boundary patch:

        path = '../sims_cil_lam/' + str(serial) + '/VTK/'+ patch + '/' + patch + '_' + str( t ) + '.vtk'
        print(path)
        mesh_cell_data = pv.read(path).cell_arrays
        vtk_list.append(mesh_cell_data[entity])

        data[entity].append(padding(np.concatenate(vtk_list) ,max))

    return data


def extract_simulation_data(serial=None):
    data = {'Ux': [], 'Uy': [], 'p': [], 'Cx': [], 'Cy': [], 'f_U': []}
    for entity in ['p', 'U_non_cons', 'Cx', 'Cy', 'f_U']:
      for time in range(int(total_times)):
        vtk_list = []
        t = round(deltat*(time*3+1)*100)/100
        if t % 1 == 0 :
          t = round(t)
        path = '../sims_cil_lam/' + str(serial) + '/VTK/'+ str(serial) + '_' + str( t ) + '.vtk'
        print(path)
        mesh_cell_data = pv.read(path).cell_arrays
        vtk_list.append(mesh_cell_data[entity])

        if entity == 'U_non_cons':
            data['Ux'].append(padding(np.concatenate(vtk_list)[:,0],max))
            data['Uy'].append(padding(np.concatenate(vtk_list)[:,1],max))

        else:
            extent = 0, 3, 0, 1
            data[entity].append(padding(np.concatenate(vtk_list), max))
    return data


directory_list = next(os.walk('../sims_cil_lam/'))[1]
num_sims_actual = len(directory_list) -1

directory_list_sim = next(os.walk('../sims_cil_lam/0/'))[1]
total_times = 100 #constant, dynamicCode, system and VTK files - no dyna,iccode

train_shape = (num_sims_actual, int(total_times) , 200000 , 6)

hdf5_path = 'dataset_unsteadyCil_fu_bound.hdf5'
hdf5_file = h5py.File(hdf5_path, mode='w')
hdf5_file.create_dataset('sim_data', train_shape, np.float32)

top_shape = (num_sims_actual, int(total_times) , 20000 , 2)
hdf5_file.create_dataset('top_bound', top_shape, np.float32)

obst_shape = (num_sims_actual, int(total_times) , 20000 , 2)
hdf5_file.create_dataset('obst_bound', obst_shape, np.float32)


count = 0 
deltat_write = [0.35,0.4,0.45,0.55,0.6,0.65,0.75,0.8,0.85,0.95]#,1.0]
mult_for_foamtovtk = [400,2000,2000,2000,2000,2000,2000,4000,4000,4000]#,4000]

for sim in tqdm(range(num_sims_actual)):
  
  deltat = deltat_write[sim]
  init_time = deltat
  mult = mult_for_foamtovtk[sim]
  max = 200000
  data = extract_simulation_data(sim)
  max = 20000
  data_top = extract_simulation_data_bound(sim,'top')
  data_obst = extract_simulation_data_bound(sim,'obstacle')

  hdf5_file['sim_data'][count, ..., 0] = data['Ux']
  hdf5_file['sim_data'][count, ..., 1] = data['Uy']
  hdf5_file['sim_data'][count, ..., 2] = data['p']
  hdf5_file['sim_data'][count, ..., 3] = data['Cx']
  hdf5_file['sim_data'][count, ..., 4] = data['Cy']
  hdf5_file['sim_data'][count, ..., 5] = data['f_U']

  hdf5_file['top_bound'][count, ..., 0] = data_top['Cx']
  hdf5_file['top_bound'][count, ..., 1] = data_top['Cy']
  hdf5_file['obst_bound'][count, ..., 0] = data_obst['Cx']
  hdf5_file['obst_bound'][count, ..., 1] = data_obst['Cy']


  count = count + 1

hdf5_file.close()
