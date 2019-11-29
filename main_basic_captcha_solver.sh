#!/bin/sh

#SBATCH -J main_basic_captcha_solver
#SBATCH -o main_basic_captcha_solver.%j.out
#SBATCH -p gpu-titanxp
#SBATCH -t 24:00:00

#SBATCH --gres=gpu:4
#SBATCH --nodelist=n8
#SBATCH --ntasks=4
#SBATCH --tasks-per-node=4
#SBATCH --cpus-per-task=1

cd  $SLURM_SUBMIT_DIR

echo "SLURM_SUBMIT_DIR=$SLURM_SUBMIT_DIR"
echo "CUDA_HOME=$CUDA_HOME"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "CUDA_VERSION=$CUDA_VERSION"

srun -l /bin/hostname
srun -l /bin/pwd
srun -l /bin/date

python3  main_basic_captcha_solver.py

date

squeue  --job  $SLURM_JOBID

echo  "##### END #####"
