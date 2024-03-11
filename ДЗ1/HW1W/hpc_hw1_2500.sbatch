#! /bin/bash

#SBATCH --job-name="HPC HW1 (N = 2500)"

#SBATCH --ntasks=1

#SBATCH --time=0-1:0

#SBATCH --mail-user=lvzaburunov@edu.hse.ru

#SBATCH --mail-type=ALL

#SBATCH --output="hpc_hw1_2500.out"

#SBATCH --error="hpc_hw1_2500.err"


# Executable

echo "Scalar implementation, opt. level O0\n\n"
#echo "OMP_NUM_THREADS=1\n"
#srun -n 1 -c 1 ./hw1_base_o0.out identity 2500
#srun -n 1 -c 1 ./hw1_base_o0.out identity 2500
#srun -n 1 -c 1 ./hw1_base_o0.out identity 2500
#srun -n 1 -c 1 ./hw1_base_o0.out identity 2500
#srun -n 1 -c 1 ./hw1_base_o0.out identity 2500

echo "\nOMP_NUM_THREADS=2\n"
srun -n 1 -c 2 ./hw1_base_o0.out identity 2500
srun -n 1 -c 2 ./hw1_base_o0.out identity 2500
srun -n 1 -c 2 ./hw1_base_o0.out identity 2500
srun -n 1 -c 2 ./hw1_base_o0.out identity 2500
srun -n 1 -c 2 ./hw1_base_o0.out identity 2500

echo "\nOMP_NUM_THREADS=4\n"
srun -n 1 -c 4 ./hw1_base_o0.out identity 2500
srun -n 1 -c 4 ./hw1_base_o0.out identity 2500
srun -n 1 -c 4 ./hw1_base_o0.out identity 2500
srun -n 1 -c 4 ./hw1_base_o0.out identity 2500
srun -n 1 -c 4 ./hw1_base_o0.out identity 2500

echo "\nOMP_NUM_THREADS=8\n"
srun -n 1 -c 8 ./hw1_base_o0.out identity 2500
srun -n 1 -c 8 ./hw1_base_o0.out identity 2500
srun -n 1 -c 8 ./hw1_base_o0.out identity 2500
srun -n 1 -c 8 ./hw1_base_o0.out identity 2500
srun -n 1 -c 8 ./hw1_base_o0.out identity 2500

echo "\nOMP_NUM_THREADS=12\n"
srun -n 1 -c 12 ./hw1_base_o0.out identity 2500
srun -n 1 -c 12 ./hw1_base_o0.out identity 2500
srun -n 1 -c 12 ./hw1_base_o0.out identity 2500
srun -n 1 -c 12 ./hw1_base_o0.out identity 2500
srun -n 1 -c 12 ./hw1_base_o0.out identity 2500

echo "\nOMP_NUM_THREADS=16\n"
srun -n 1 -c 16 ./hw1_base_o0.out identity 2500
srun -n 1 -c 16 ./hw1_base_o0.out identity 2500
srun -n 1 -c 16 ./hw1_base_o0.out identity 2500
srun -n 1 -c 16 ./hw1_base_o0.out identity 2500
srun -n 1 -c 16 ./hw1_base_o0.out identity 2500

echo "\nOMP_NUM_THREADS=24\n"
srun -n 1 -c 24 ./hw1_base_o0.out identity 2500
srun -n 1 -c 24 ./hw1_base_o0.out identity 2500
srun -n 1 -c 24 ./hw1_base_o0.out identity 2500
srun -n 1 -c 24 ./hw1_base_o0.out identity 2500
srun -n 1 -c 24 ./hw1_base_o0.out identity 2500


echo "\nScalar implementation, opt. level O3\n"
#echo "OMP_NUM_THREADS=1\n"
#srun -n 1 -c 1 ./hw1_base_o3.out identity 2500
#srun -n 1 -c 1 ./hw1_base_o3.out identity 2500
#srun -n 1 -c 1 ./hw1_base_o3.out identity 2500
#srun -n 1 -c 1 ./hw1_base_o3.out identity 2500
#srun -n 1 -c 1 ./hw1_base_o3.out identity 2500

echo "\nOMP_NUM_THREADS=2\n"
srun -n 1 -c 2 ./hw1_base_o3.out identity 2500
srun -n 1 -c 2 ./hw1_base_o3.out identity 2500
srun -n 1 -c 2 ./hw1_base_o3.out identity 2500
srun -n 1 -c 2 ./hw1_base_o3.out identity 2500
srun -n 1 -c 2 ./hw1_base_o3.out identity 2500

echo "\nOMP_NUM_THREADS=4\n"
srun -n 1 -c 4 ./hw1_base_o3.out identity 2500
srun -n 1 -c 4 ./hw1_base_o3.out identity 2500
srun -n 1 -c 4 ./hw1_base_o3.out identity 2500
srun -n 1 -c 4 ./hw1_base_o3.out identity 2500
srun -n 1 -c 4 ./hw1_base_o3.out identity 2500

echo "\nOMP_NUM_THREADS=8\n"
srun -n 1 -c 8 ./hw1_base_o3.out identity 2500
srun -n 1 -c 8 ./hw1_base_o3.out identity 2500
srun -n 1 -c 8 ./hw1_base_o3.out identity 2500
srun -n 1 -c 8 ./hw1_base_o3.out identity 2500
srun -n 1 -c 8 ./hw1_base_o3.out identity 2500

echo "\nOMP_NUM_THREADS=12\n"
srun -n 1 -c 12 ./hw1_base_o3.out identity 2500
srun -n 1 -c 12 ./hw1_base_o3.out identity 2500
srun -n 1 -c 12 ./hw1_base_o3.out identity 2500
srun -n 1 -c 12 ./hw1_base_o3.out identity 2500
srun -n 1 -c 12 ./hw1_base_o3.out identity 2500

echo "\nOMP_NUM_THREADS=16\n"
srun -n 1 -c 16 ./hw1_base_o3.out identity 2500
srun -n 1 -c 16 ./hw1_base_o3.out identity 2500
srun -n 1 -c 16 ./hw1_base_o3.out identity 2500
srun -n 1 -c 16 ./hw1_base_o3.out identity 2500
srun -n 1 -c 16 ./hw1_base_o3.out identity 2500

echo "\nOMP_NUM_THREADS=24\n"
srun -n 1 -c 24 ./hw1_base_o3.out identity 2500
srun -n 1 -c 24 ./hw1_base_o3.out identity 2500
srun -n 1 -c 24 ./hw1_base_o3.out identity 2500
srun -n 1 -c 24 ./hw1_base_o3.out identity 2500
srun -n 1 -c 24 ./hw1_base_o3.out identity 2500

echo "\nAVX512 implementation, opt. level O0\n"
echo "OMP_NUM_THREADS=1\n"
#srun -n 1 -c 1 ./hw1_avx_o0.out identity 2500
#srun -n 1 -c 1 ./hw1_avx_o0.out identity 2500
#srun -n 1 -c 1 ./hw1_avx_o0.out identity 2500
#srun -n 1 -c 1 ./hw1_avx_o0.out identity 2500
#srun -n 1 -c 1 ./hw1_avx_o0.out identity 2500

echo "\nOMP_NUM_THREADS=2\n"
srun -n 1 -c 2 ./hw1_avx_o0.out identity 2500
srun -n 1 -c 2 ./hw1_avx_o0.out identity 2500
srun -n 1 -c 2 ./hw1_avx_o0.out identity 2500
srun -n 1 -c 2 ./hw1_avx_o0.out identity 2500
srun -n 1 -c 2 ./hw1_avx_o0.out identity 2500

echo "\nOMP_NUM_THREADS=4\n"
srun -n 1 -c 4 ./hw1_avx_o0.out identity 2500
srun -n 1 -c 4 ./hw1_avx_o0.out identity 2500
srun -n 1 -c 4 ./hw1_avx_o0.out identity 2500
srun -n 1 -c 4 ./hw1_avx_o0.out identity 2500
srun -n 1 -c 4 ./hw1_avx_o0.out identity 2500

echo "\nOMP_NUM_THREADS=8\n"
srun -n 1 -c 8 ./hw1_avx_o0.out identity 2500
srun -n 1 -c 8 ./hw1_avx_o0.out identity 2500
srun -n 1 -c 8 ./hw1_avx_o0.out identity 2500
srun -n 1 -c 8 ./hw1_avx_o0.out identity 2500
srun -n 1 -c 8 ./hw1_avx_o0.out identity 2500

echo "\nOMP_NUM_THREADS=12\n"
srun -n 1 -c 12 ./hw1_avx_o0.out identity 2500
srun -n 1 -c 12 ./hw1_avx_o0.out identity 2500
srun -n 1 -c 12 ./hw1_avx_o0.out identity 2500
srun -n 1 -c 12 ./hw1_avx_o0.out identity 2500
srun -n 1 -c 12 ./hw1_avx_o0.out identity 2500

echo "\nOMP_NUM_THREADS=16\n"
srun -n 1 -c 16 ./hw1_avx_o0.out identity 2500
srun -n 1 -c 16 ./hw1_avx_o0.out identity 2500
srun -n 1 -c 16 ./hw1_avx_o0.out identity 2500
srun -n 1 -c 16 ./hw1_avx_o0.out identity 2500
srun -n 1 -c 16 ./hw1_avx_o0.out identity 2500

echo "\nOMP_NUM_THREADS=24\n"
srun -n 1 -c 24 ./hw1_avx_o0.out identity 2500
srun -n 1 -c 24 ./hw1_avx_o0.out identity 2500
srun -n 1 -c 24 ./hw1_avx_o0.out identity 2500
srun -n 1 -c 24 ./hw1_avx_o0.out identity 2500
srun -n 1 -c 24 ./hw1_avx_o0.out identity 2500


echo "\nAVX512 implementation, opt. level O3\n"
echo "OMP_NUM_THREADS=1\n"
#srun -n 1 -c 1 ./hw1_avx_o3.out identity 2500
#srun -n 1 -c 1 ./hw1_avx_o3.out identity 2500
#srun -n 1 -c 1 ./hw1_avx_o3.out identity 2500
#srun -n 1 -c 1 ./hw1_avx_o3.out identity 2500
#srun -n 1 -c 1 ./hw1_avx_o3.out identity 2500

echo "\nOMP_NUM_THREADS=2\n"
srun -n 1 -c 2 ./hw1_avx_o3.out identity 2500
srun -n 1 -c 2 ./hw1_avx_o3.out identity 2500
srun -n 1 -c 2 ./hw1_avx_o3.out identity 2500
srun -n 1 -c 2 ./hw1_avx_o3.out identity 2500
srun -n 1 -c 2 ./hw1_avx_o3.out identity 2500

echo "\nOMP_NUM_THREADS=4\n"
srun -n 1 -c 4 ./hw1_avx_o3.out identity 2500
srun -n 1 -c 4 ./hw1_avx_o3.out identity 2500
srun -n 1 -c 4 ./hw1_avx_o3.out identity 2500
srun -n 1 -c 4 ./hw1_avx_o3.out identity 2500
srun -n 1 -c 4 ./hw1_avx_o3.out identity 2500

echo "\nOMP_NUM_THREADS=8\n"
srun -n 1 -c 8 ./hw1_avx_o3.out identity 2500
srun -n 1 -c 8 ./hw1_avx_o3.out identity 2500
srun -n 1 -c 8 ./hw1_avx_o3.out identity 2500
srun -n 1 -c 8 ./hw1_avx_o3.out identity 2500
srun -n 1 -c 8 ./hw1_avx_o3.out identity 2500

echo "\nOMP_NUM_THREADS=12\n"
srun -n 1 -c 12 ./hw1_avx_o3.out identity 2500
srun -n 1 -c 12 ./hw1_avx_o3.out identity 2500
srun -n 1 -c 12 ./hw1_avx_o3.out identity 2500
srun -n 1 -c 12 ./hw1_avx_o3.out identity 2500
srun -n 1 -c 12 ./hw1_avx_o3.out identity 2500

echo "\nOMP_NUM_THREADS=16\n"
srun -n 1 -c 16 ./hw1_avx_o3.out identity 2500
srun -n 1 -c 16 ./hw1_avx_o3.out identity 2500
srun -n 1 -c 16 ./hw1_avx_o3.out identity 2500
srun -n 1 -c 16 ./hw1_avx_o3.out identity 2500
srun -n 1 -c 16 ./hw1_avx_o3.out identity 2500

echo "\nOMP_NUM_THREADS=24\n"
srun -n 1 -c 24 ./hw1_avx_o3.out identity 2500
srun -n 1 -c 24 ./hw1_avx_o3.out identity 2500
srun -n 1 -c 24 ./hw1_avx_o3.out identity 2500
srun -n 1 -c 24 ./hw1_avx_o3.out identity 2500
srun -n 1 -c 24 ./hw1_avx_o3.out identity 2500
