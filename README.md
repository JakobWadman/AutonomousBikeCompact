# AutonomousBikeCompact

This repository is intended to provide the necessary files to continue the project Reinforcement learning of a nonlinear bike
controller (fall 2019).

The pdf-file instructions contain various insights we aquired during the project that may be beneficial.

The pdf-file report is our report for the project.

The directory RL_directory contains the material related to reinforcement learning. It contains:
* barebone_setup - directory with files for training an RL-agent on a linear bike model. These files are as striped down as possible to provide a template for when you want to modify the setup. The content consist of:
  + simple_ppo_notebook - the main script in the form of a jupyter notebook
  + Bike_barebone - the environment with the linear model
  + utility_functions - contains function for comparing the RL-agent to an LQR
  + multiprocessing_env - script that makes it possible to run environments in parallel
  + saved_networks - folder where the networks are saved. 
* final_version_with_saturation - directory with files for training an RL-agent on the linear model, but with the addition of saturation on the steering angle rate. The files has a bit more features than the ones in barebones folder. For example there is a simple render of the bike included in the environment file.
