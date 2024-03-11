#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "stdio.h"
#include "chrono"
#include "output.h"
//#pragma once
/*#ifdef __INTELLISENSE__
void __syncthreads();
#endif*/
//#ifndef __CUDACC__
//#define __CUDACC__
//#endif
#include "device_functions.h"


void cuda_mm(int M, int N, int K, double *A, double *B, double *&C);

