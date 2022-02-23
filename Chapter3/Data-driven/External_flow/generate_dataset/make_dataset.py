import os
from tqdm import *
import subprocess
import random

num_runs = 200
x_dim = random.sample(range(50, 1000), num_runs)
y_dim = random.sample(range(100,500), num_runs) #remember that this is only 1/2

x_dim = [round(x2 / 1000, 2) for x2 in x_dim]
y_dim = [round(y / 1000, 2) for y in y_dim]

start=50
for i in tqdm(range(num_runs)):
    with open(os.devnull, 'w') as devnull:
        # Remove any previous simulation file
        cmd = "rm -rf simulation_data/" + str(i+start)
        subprocess.check_call(cmd.split(' '), stdout=devnull, stderr=devnull)

        # Copy the OpenFOAM forwardStep directory
        cmd = "cp -a ./original/. ./simulation_data/" + str(i+start)
        subprocess.check_call(cmd.split(' '), stdout=devnull, stderr=devnull)

        # Remove the blockMeshDict file from system directory
        cmd = "rm -f ./simulation_data/" + str(i+start) + "/system/blockMeshDict"
        subprocess.check_call(cmd.split(' '), stdout=devnull, stderr=devnull)

        # Execute python program to write a blockMeshDict file
        cmd = "python gen_blockMeshDict.py" + " " + str(x_dim[i]) + " " + str(y_dim[i]) 
        subprocess.check_call(cmd.split(' '), stdout=devnull, stderr=devnull)

        # Move the blockMeshDict file to system directory
        cmd = "mv blockMeshDict ./simulation_data/" + str(i+start) + "/system"
        subprocess.check_call(cmd.split(' '), stdout=devnull, stderr=devnull)

