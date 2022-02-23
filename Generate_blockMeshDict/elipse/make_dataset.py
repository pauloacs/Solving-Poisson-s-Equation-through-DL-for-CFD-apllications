import os
from tqdm import *
import subprocess
import random

num_runs = 150
a = random.sample(range(50, 1200), num_runs)
b = random.sample(range(50, 300), num_runs)

a = [round(x / 1000, 2) for x in a]
b = [round(x / 1000, 2) for x in b]

for i in range(len(a)):
    if b[i] > a[i]:
      b[i] = a[i]

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
        cmd = "python elipse_new.py" + " " + str(a[i]) + " " + str(b[i])
        subprocess.check_call(cmd.split(' '), stdout=devnull, stderr=devnull)

        # Move the blockMeshDict file to system directory
        cmd = "mv blockMeshDict ./simulation_data/" + str(i) + "/system"
        subprocess.check_call(cmd.split(' '), stdout=devnull, stderr=devnull)
       
