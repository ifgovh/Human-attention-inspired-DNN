#!/bin/bash
#PBS -P cortical
#PBS -N simple_python
#PBS -q defaultQ 
#PBS -l select=1:ncpus=2:mem=2gb
#PBS -l walltime=23:59:59
#PBS -e PBSout_GPU/P_err_av
#PBS -o PBSout_GPU/P_out_av
#PBS -J 1-3

#module load python/3.6.5
cd ~
source tf/bin/activate
cd "$PBS_O_WORKDIR"
params='sed "${PBS_ARRAY_INDEX}q;d" job_params'
param_array=( $params )
python3 main.py --loc_hidden=192 --hidden_size=320 --loss_fun_action=${param_array[0]} --loss_fun_baseline=${param_array[1]} --dataset_name='CIFAR' --patch_size=10 --epochs=2000 --train_patience=1000 --PBSarray_ID=${PBS_ARRAY_INDEX}

#--batch_szie= --loc_hidden=192 --hidden_size=320 --glimpse_hidden= --num_glimpse= --glimpse_scale= --loss_fun_action= --loss_fun_baseline= 