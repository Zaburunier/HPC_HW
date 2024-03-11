#! /bin/bash
#SBATCH --job-name="HW2 (N = 2500)"
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --time=0-1:0
#SBATCH --mail-user=lvzaburunov@edu.hse.ru
#SBATCH --mail-type=ALL
#SBATCH --output="hw2-2500".out
#SBATCH --error="hw2-2500".err
module load nvidia_sdk/nvhpc/21.9

echo "################################\n"
echo "CUDA naive implementation, opt. level O0\n"
echo "################################\n\n"

srun ./hw2_1_o0.out identity 2500
srun ./hw2_1_o0.out identity 2500
srun ./hw2_1_o0.out identity 2500
srun ./hw2_1_o0.out identity 2500
srun ./hw2_1_o0.out identity 2500

echo "################################\n"
echo "CUDA pinned memory implementation, opt. level O0\n"
echo "################################\n\n"

srun ./hw2_2_o0.out identity 2500
srun ./hw2_2_o0.out identity 2500
srun ./hw2_2_o0.out identity 2500
srun ./hw2_2_o0.out identity 2500
srun ./hw2_2_o0.out identity 2500

echo "################################\n"
echo "CUDA streams implementation, opt. level O0\n"
echo "################################\n\n"

echo "---1 STREAM---n"
srun ./hw2_31_o0.out identity 2500
srun ./hw2_31_o0.out identity 2500
srun ./hw2_31_o0.out identity 2500
srun ./hw2_31_o0.out identity 2500
srun ./hw2_31_o0.out identity 2500

echo "---2 STREAMS---n"
srun ./hw2_32_o0.out identity 2500
srun ./hw2_32_o0.out identity 2500
srun ./hw2_32_o0.out identity 2500
srun ./hw2_32_o0.out identity 2500
srun ./hw2_32_o0.out identity 2500

echo "---4 STREAMS---n"
srun ./hw2_34_o0.out identity 2500
srun ./hw2_34_o0.out identity 2500
srun ./hw2_34_o0.out identity 2500
srun ./hw2_34_o0.out identity 2500
srun ./hw2_34_o0.out identity 2500

echo "---8 STREAMS---n"
srun ./hw2_38_o0.out identity 2500
srun ./hw2_38_o0.out identity 2500
srun ./hw2_38_o0.out identity 2500
srun ./hw2_38_o0.out identity 2500
srun ./hw2_38_o0.out identity 2500

echo "################################\n"
echo "CUDA tiled implementation, opt. level O0\n"
echo "################################\n\n"

srun ./hw2_41_o0.out identity 2500
srun ./hw2_41_o0.out identity 2500
srun ./hw2_41_o0.out identity 2500
srun ./hw2_41_o0.out identity 2500
srun ./hw2_41_o0.out identity 2500

echo "################################\n"
echo "CUDA tiled implementation with thread combining, opt. level O0\n"
echo "################################\n\n"

srun ./hw2_44_o0.out identity 2500
srun ./hw2_44_o0.out identity 2500
srun ./hw2_44_o0.out identity 2500
srun ./hw2_44_o0.out identity 2500
srun ./hw2_44_o0.out identity 2500