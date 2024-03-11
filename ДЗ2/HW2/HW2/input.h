#pragma once
#include "stdio.h"
#include "fstream"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"


double* RowMajorToColumnMajor(double*, int, int);


double* GetMatrixFromFile(char*, int&, int&, bool = false);


double* GetMatrixFromConsoleInput(int&, int&, bool = false);


double* GetRandomMatrix(int&, int&, double = 0.0, double = 1.0, bool = false);


double *GetIdentityMatrix(int, bool = false);