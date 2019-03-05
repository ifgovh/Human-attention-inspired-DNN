#!/bin/bash
#PBS -P cortical
#PBS -N simple_python
#PBS -q defaultQ 
#PBS -l select=1:ncpus=2:mem=16gb
#PBS -l walltime=23:59:59
#PBS -e PBSout_GPU/P_err_av
#PBS -o PBSout_GPU/P_out_av
##PBS -J 1-5

#module load python/3.6.5
cd ~
source tf/bin/activate
cd "$PBS_O_WORKDIR"
python3 main.py --loc_hidden=192 --hidden_size=320 --dataset_name='CIFAR' --patch_size=10 --epochs=2000

