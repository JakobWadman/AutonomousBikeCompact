# AutonomousBikeCompact

This repository is intended to provide the necessary files to continue the project Reinforcement learning of a nonlinear bike
controller (fall 2019).

The pdf-file instructions contain various insights we aquired during the project that may be beneficial.

The pdf-file report is our report for the project.

The directory RL_directory contains the material related to reinforcement learning. It contains:
* barebone_setup - directory with files for training an RL-agent on a linear bike model. These files are as striped down as possible to provide a template for when you want to modify the setup. The content consists of:
  + simple_ppo_notebook - the main script in the form of a jupyter notebook
  + Bike_barebone - the environment file with the linear model and a reward based on the cost that an LQR optimizes
  + utility_functions - contains functions for comparing the RL-agent to an LQR
  + multiprocessing_env - script that makes it possible to run environments in parallel
  + saved_networks - directory where the networks are saved. 
* final_version_with_saturation - directory with files for training an RL-agent on the linear model, but with the addition of saturation on the steering angle rate. The files has a bit more features than the ones in barebones directory. For example there is a simple render of the bike included in the environment file. The content consists of:
  + ppo_4states - the main script in the form of a jupyter notebook
  + Bike_4states -  the environment file with the saturated steering angle
  + utility_functions_4states - contains functions for comparing the RL-agent to an LQR
  + test_bike_env_4states - plots the state trajectories for the enviroment when using LQR feedback and prints the cumulative reward for the trajectory.
  + multiprocessing_env - script that makes it possible to run environments in parallel
  + saved_networks - - directory where the networks are saved. 
 

The directory Matlab_scripts contain the following files and directories:
* get_A_B _and_K_matrices_from_tf - script for calculating the discrete state-space matrices as well as the LQR gain from the transferfunction for the bike that we used during the project
* Nonlinearmodel - directory with files for simulating a nonlinear model of a bike controlled by an LQR, developed using a linearized version of the model.
