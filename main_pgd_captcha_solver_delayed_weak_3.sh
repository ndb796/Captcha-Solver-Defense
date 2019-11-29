#!/bin/sh

#SBATCH -J main_pgd_captcha_solver_delayed_weak_3
#SBATCH -o main_pgd_captcha_solver_delayed_weak_3.%j.out
#SBATCH -p gpu-titanxp
#SBATCH -t 24:00:00

#SBATCH --gres=gpu:4
#SBATCH --nodelist=n7
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

python3 main_pgd_captcha_solver_delayed_weak_3.py

date

squeue --job $SLURM_JOBID

echo "##### END #####"

