#!/bin/bash
#PBS -P cortical
#PBS -N find_super_params
#PBS -q defaultQ 
#PBS -l select=1:ncpus=1:mem=12gb
#PBS -l walltime=150:59:59
#PBS -e PBSout_GPU/
#PBS -o PBSout_GPU/
##PBS -J 1-30

#module load python/3.6.5
SOURCE_CODE="find_super_params" 
cd ~
source tf/bin/activate
cd "$PBS_O_WORKDIR"
# give params via PBSarrayID
params=`sed "${PBS_ARRAY_INDEX}q;d" job_params`
param_array=( $params )
# save log file for the source code
pfile=`find . -name "${SOURCE_CODE}.py"`;
SOURCE_FILE=$(echo ${pfile});
SOURCE_FILENAME=`basename ${pfile}`;
A=`date '+%s'`
cp ${SOURCE_FILE} ${PBS_O_WORKDIR}/${A}-${SOURCE_FILENAME}.log;
# run the code
python3 "${SOURCE_CODE}.py"