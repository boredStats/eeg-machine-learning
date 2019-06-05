#!/bin/bash

#SBATCH -J EEG graph theory               # Job name
#SBATCH -o job.%j.out         # Name of stdout output file (%j expands to jobId)
#SBATCH -N 1                  # Total number of nodes requested
#SBATCH -n 16                  # Total number of mpi tasks requested
#SBATCH -t 48:00:00           # Run time (hh:mm:ss)
#SBATCH -p clint           # Queue

module load anaconda3
python ./../eeg_networks.py