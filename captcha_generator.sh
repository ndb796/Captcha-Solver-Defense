#!/bin/sh

#SBATCH -J captcha_generator
#SBATCH -o captcha_generator.%j.out
#SBATCH -p cpu-i7
#SBATCH -t 24:00:00

cd  $SLURM_SUBMIT_DIR

echo "SLURM_SUBMIT_DIR=$SLURM_SUBMIT_DIR"
echo "CUDA_HOME=$CUDA_HOME"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "CUDA_VERSION=$CUDA_VERSION"

srun -l /bin/hostname
srun -l /bin/pwd
srun -l /bin/date

python3 captcha_generator.py

date

squeue --job $SLURM_JOBID

echo "##### END #####"
