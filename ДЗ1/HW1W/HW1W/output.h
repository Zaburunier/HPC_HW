#pragma once
#include "stdio.h"


bool AreEqual(double*, double*, int, float = 1e-04f);


void PrintMatrix(double*, int, int);


void PrintArray(double*, int);


void WriteMatrixToFile(double*, char*, int&, int&);