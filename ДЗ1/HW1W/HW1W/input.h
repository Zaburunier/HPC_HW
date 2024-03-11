#pragma once
#include "stdio.h"
#include "fstream"


double* RowMajorToColumnMajor(double*, int, int);


double* GetMatrixFromFile(char*, int&, int&);


double* GetMatrixFromConsoleInput(int&, int&);


double* GetRandomMatrix(int&, int&, double = 0.0, double = 1.0);


double *GetIdentityMatrix(int);