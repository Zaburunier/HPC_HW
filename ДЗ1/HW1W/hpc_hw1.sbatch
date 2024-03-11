#! /bin/bash

#SBATCH --job-name="HPC HW1 (`$1` threads)"

#SBATCH --ntasks=`$1`

#SBATCH --cpus-per-task=1

#SBATCH --time=0-1:0

#SBATCH --mail-user=lvzaburunov@edu.hse.ru

#SBATCH --mail-type=ALL

#SBATCH --output="hpc_hw1_2500.out"

#SBATCH --error="hpc_hw1_2500.err"


# Executable

echo -e "Scalar implementation, opt. level O0\n"
echo -e "N = 400\n"
srun ./hw1_base_o0.out identity 400
srun ./hw1_base_o0.out identity 400
srun ./hw1_base_o0.out identity 400
srun ./hw1_base_o0.out identity 400
srun ./hw1_base_o0.out identity 400

echo -e "\nN = 1000\n"
srun ./hw1_base_o0.out identity 1000
srun ./hw1_base_o0.out identity 1000
srun ./hw1_base_o0.out identity 1000
srun ./hw1_base_o0.out identity 1000
srun ./hw1_base_o0.out identity 1000

echo -e "\nN = 2500\n"
srun ./hw1_base_o0.out identity 2500
srun ./hw1_base_o0.out identity 2500
srun ./hw1_base_o0.out identity 2500
srun ./hw1_base_o0.out identity 2500
srun ./hw1_base_o0.out identity 2500


echo "\nScalar implementation, opt. level O3\n"
echo -e "N = 400\n"
srun ./hw1_base_o3.out identity 400
srun ./hw1_base_o3.out identity 400
srun ./hw1_base_o3.out identity 400
srun ./hw1_base_o3.out identity 400
srun ./hw1_base_o3.out identity 400

echo -e "\nN = 1000\n"
srun ./hw1_base_o3.out identity 1000
srun ./hw1_base_o3.out identity 1000
srun ./hw1_base_o3.out identity 1000
srun ./hw1_base_o3.out identity 1000
srun ./hw1_base_o3.out identity 1000

echo -e "\nN = 2500\n"
srun ./hw1_base_o3.out identity 2500
srun ./hw1_base_o3.out identity 2500
srun ./hw1_base_o3.out identity 2500
srun ./hw1_base_o3.out identity 2500
srun ./hw1_base_o3.out identity 2500


echo -e "AVX512 implementation, opt. level O0\n"
echo -e "N = 400\n"
srun ./hw1_avx_o0.out identity 400
srun ./hw1_avx_o0.out identity 400
srun ./hw1_avx_o0.out identity 400
srun ./hw1_avx_o0.out identity 400
srun ./hw1_avx_o0.out identity 400

echo -e "\nN = 1000\n"
srun ./hw1_avx_o0.out identity 1000
srun ./hw1_avx_o0.out identity 1000
srun ./hw1_avx_o0.out identity 1000
srun ./hw1_avx_o0.out identity 1000
srun ./hw1_avx_o0.out identity 1000

echo -e "\nN = 2500\n"
srun ./hw1_avx_o0.out identity 2500
srun ./hw1_avx_o0.out identity 2500
srun ./hw1_avx_o0.out identity 2500
srun ./hw1_avx_o0.out identity 2500
srun ./hw1_avx_o0.out identity 2500


echo "\AVX512 implementation, opt. level O3\n"
echo -e "N = 400\n"
srun ./hw1_avx_o3.out identity 400
srun ./hw1_avx_o3.out identity 400
srun ./hw1_avx_o3.out identity 400
srun ./hw1_avx_o3.out identity 400
srun ./hw1_avx_o3.out identity 400

echo -e "\nN = 1000\n"
srun ./hw1_avx_o3.out identity 1000
srun ./hw1_avx_o3.out identity 1000
srun ./hw1_avx_o3.out identity 1000
srun ./hw1_avx_o3.out identity 1000
srun ./hw1_avx_o3.out identity 1000

echo -e "\nN = 2500\n"
srun ./hw1_avx_o3.out identity 2500
srun ./hw1_avx_o3.out identity 2500
srun ./hw1_avx_o3.out identity 2500
srun ./hw1_avx_o3.out identity 2500
srun ./hw1_avx_o3.out identity 2500

