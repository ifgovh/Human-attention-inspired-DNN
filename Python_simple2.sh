#!/bin/bash
#PBS -P cortical
#PBS -N find_super_params
#PBS -q defaultQ 
#PBS -l select=1:ncpus=12:mem=32gb
#PBS -l walltime=150:59:59
#PBS -e PBSout_GPU/
#PBS -o PBSout_GPU/
##PBS -J 1-30

#module load python/3.6.5
cd ~
source tf/bin/activate
cd "$PBS_O_WORKDIR"
params=`sed "${PBS_ARRAY_INDEX}q;d" job_params`
param_array=( $params )
python3 find_super_params_cpu.py

