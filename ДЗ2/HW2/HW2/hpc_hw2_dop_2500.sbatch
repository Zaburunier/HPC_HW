#! /bin/bash
#SBATCH --job-name="HW2 DOP (N = 2500)"
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --time=0-1:0
#SBATCH --mail-user=lvzaburunov@edu.hse.ru
#SBATCH --mail-type=ALL
#SBATCH --output="hw2-dop-2500".out
#SBATCH --error="hw2-dop-2500".err
module load nvidia_sdk/nvhpc/21.9

echo "################################\n"
echo "BLOCK SIZE = 2\n"
echo "################################\n\n"

srun ./hw2_dop_2.out identity 2500
srun ./hw2_dop_2.out identity 2500
srun ./hw2_dop_2.out identity 2500
srun ./hw2_dop_2.out identity 2500
srun ./hw2_dop_2.out identity 2500

echo "################################\n"
echo "BLOCK SIZE = 4\n"
echo "################################\n\n"

srun ./hw2_dop_4.out identity 2500
srun ./hw2_dop_4.out identity 2500
srun ./hw2_dop_4.out identity 2500
srun ./hw2_dop_4.out identity 2500
srun ./hw2_dop_4.out identity 2500

echo "################################\n"
echo "BLOCK SIZE = 8\n"
echo "################################\n\n"

srun ./hw2_dop_8.out identity 2500
srun ./hw2_dop_8.out identity 2500
srun ./hw2_dop_8.out identity 2500
srun ./hw2_dop_8.out identity 2500
srun ./hw2_dop_8.out identity 2500

echo "################################\n"
echo "BLOCK SIZE = 12\n"
echo "################################\n\n"

srun ./hw2_dop_12.out identity 2500
srun ./hw2_dop_12.out identity 2500
srun ./hw2_dop_12.out identity 2500
srun ./hw2_dop_12.out identity 2500
srun ./hw2_dop_12.out identity 2500

echo "################################\n"
echo "BLOCK SIZE = 16\n"
echo "################################\n\n"

srun ./hw2_dop_16.out identity 2500
srun ./hw2_dop_16.out identity 2500
srun ./hw2_dop_16.out identity 2500
srun ./hw2_dop_16.out identity 2500
srun ./hw2_dop_16.out identity 2500

echo "################################\n"
echo "BLOCK SIZE = 24\n"
echo "################################\n\n"

srun ./hw2_dop_24.out identity 2500
srun ./hw2_dop_24.out identity 2500
srun ./hw2_dop_24.out identity 2500
srun ./hw2_dop_24.out identity 2500
srun ./hw2_dop_24.out identity 2500

echo "################################\n"
echo "BLOCK SIZE = 32\n"
echo "################################\n\n"

srun ./hw2_dop_32.out identity 2500
srun ./hw2_dop_32.out identity 2500
srun ./hw2_dop_32.out identity 2500
srun ./hw2_dop_32.out identity 2500
srun ./hw2_dop_32.out identity 2500