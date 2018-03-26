#!/bin/bash
#set a job name  
#SBATCH --job-name=jawrob50
#################
#a file for job output, you can check job progress
#SBATCH --output=output.out
#################
# a file for errors from the job
#SBATCH --error=error.err
#################
#time you think you need; default is one day
#in minutes in this case, hh:mm:ss
#SBATCH --time=24:00:00
#################
#number of tasks you are requesting
#SBATCH -N 1
#SBATCH --exclusive
#################
#partition to use
#SBATCH --partition=ser-par-10g-4
#################

source activate bci_env
python ../../signal_model/offline_analysis_m.py $1
