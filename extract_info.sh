#!/bin/bash
#PBS -P cortical
#PBS -N extract_info
#PBS -q defaultQ 
#PBS -l select=1:ncpus=1:mem=8gb
#PBS -l walltime=00:19:59
#PBS -e PBSout_GPU/
#PBS -o PBSout_GPU/
##PBS -J 1-30

#module load python/3.6.5
cd ~
source tf/bin/activate
cd "$PBS_O_WORKDIR"
python3 extract_info_best_model.py

