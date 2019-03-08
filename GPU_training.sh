#!/bin/bash
#PBS -P cortical
#PBS -N GPU_training
#PBS -q defaultQ 
#PBS -l select=1:ncpus=1:ngpus=1:mem=4gb
#PBS -l walltime=3:59:59
#PBS -e PBSout_GPU/
#PBS -o PBSout_GPU/
#PBS -J 1-8

#module load python/3.6.5
cd ~
source tf/bin/activate
cd "$PBS_O_WORKDIR"
params=`sed "${PBS_ARRAY_INDEX}q;d" job_params_gpu`
param_array=( $params )
python3 main.py --loc_hidden=256 --glimpse_hidden=${param_array[0]} --hidden_size=${param_array[1]} --batch_size=256 --num_glimpse=6\
 --weight_decay=0  --dataset_name='CIFAR' --patch_size=${param_array[2]} --epochs=2000 --train_patience=1000\
  --PBSarray_ID=${PBS_ARRAY_INDEX} --use_gpu=True

#--batch_szie= --loc_hidden=192 --hidden_size=320 --glimpse_hidden= --num_glimpse= --glimpse_scale= --loss_fun_action= --loss_fun_baseline= 
