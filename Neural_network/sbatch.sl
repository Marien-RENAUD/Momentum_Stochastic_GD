#!/bin/bash

# Your job name (displayed by the queue)
#SBATCH -J jherm
#SBATCH --output=slurm.out
#SBATCH -C "sirocco&a100"
# walltime (hh:mm::ss)
#SBATCH -t 24:00:00

# Specify the number of nodes(nodes=) and the number of cores per nodes(tasks-pernode=) to be used

#SBATCH --ntasks-per-node=1

 

# change working directory
# SBATCH --chdir=.

# fin des directives PBS
#############################

module load language/python/3.9.6

srun convexity_measure_run.sh
echo "Job finished" 