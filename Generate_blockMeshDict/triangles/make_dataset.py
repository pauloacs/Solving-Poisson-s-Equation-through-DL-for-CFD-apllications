import os
from tqdm import *
import subprocess
import random

num_runs = 15

a_list = []
b_list = []
scale = 4
grading = 4

a_ = [0.5, 0.75, 1.0, 1.25, 1.5]
b_ = [0.5, 1.0, 1.5]

for a in a_:
  for b in b_:
    a_list.append(a/2)    
    b_list.append(b)

print(a_list) 

for i in tqdm(range(num_runs)):
    with open(os.devnull, 'w') as devnull:
        # Remove any previous simulation file
        cmd = "rm -rf simulation_data/" + str(i)
        subprocess.check_call(cmd.split(' '), stdout=devnull, stderr=devnull)

        # Copy the OpenFOAM forwardStep directory
        cmd = "cp -a ./original/. ./simulation_data/" + str(i)
        subprocess.check_call(cmd.split(' '), stdout=devnull, stderr=devnull)

        # Remove the blockMeshDict file from system directory
        cmd = "rm -f ./simulation_data/" + str(i) + "/system/blockMeshDict"
        subprocess.check_call(cmd.split(' '), stdout=devnull, stderr=devnull)

        # Execute python program to write a blockMeshDict file
        cmd = "python gen_blockMeshDict.py" + " 4 " + str(b_list[i] + 4.) + " " + str(a_list[i]) + " " + str(scale) + " " + str(grading)
        subprocess.check_call(cmd.split(' '), stdout=devnull, stderr=devnull)

        # Move the blockMeshDict file to system directory
        cmd = "mv blockMeshDict ./simulation_data/" + str(i) + "/system"
        subprocess.check_call(cmd.split(' '), stdout=devnull, stderr=devnull)
