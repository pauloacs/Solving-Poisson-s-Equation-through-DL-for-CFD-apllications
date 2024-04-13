import os
from tqdm import *
import subprocess
import random

num_runs = 50

y_max = []
for i in range(10):
   y_max.append(500)
for i in range(10):
   y_max.append(600)
for i in range(10):
   y_max.append(750)
for i in range(10):
   y_max.append(900)
for i in range(10):
   y_max.append(1000)


#y_max = random.sample(range(500, 1000), num_runs)


r_int = []
for y in y_max:
      if  y > 800:
         r_int.append(random.randint(350, int(y/2)))
      elif 700 <= y < 800:
         r_int.append(random.randint(300, int(y/2)))
      else:
         r_int.append(random.randint(200, int(y/2)))
#r_int = random.sample(range(100, 210), num_runs)
#x_min = random.sample(range(1000, 2000), num_runs)

#x_min = -1.0
y_max = [round(x / 1000, 2) for x in y_max]
r_int = [round(x / 1000, 2) for x in r_int]

#x_min = [round(-x2 / 1000, 2) for x2 in x_min]





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
        cmd = "python gen_blockMeshDict.py" + " " + str(r_int[i]) + " " + str(y_max[i])
        subprocess.check_call(cmd.split(' '), stdout=devnull, stderr=devnull)

        # Move the blockMeshDict file to system directory
        cmd = "mv blockMeshDict ./simulation_data/" + str(i) + "/system"
        subprocess.check_call(cmd.split(' '), stdout=devnull, stderr=devnull)
