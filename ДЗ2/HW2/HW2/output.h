#pragma once
#include "stdio.h"
#include "chrono"


bool AreEqual(double*, double*, int, float = 1e-04f);


void PrintMatrix(double*, int, int);


void PrintArray(double*, int);


void WriteMatrixToFile(double*, char*, int&, int&);


double ConvertChronoToSeconds(std::chrono::system_clock::time_point, std::chrono::system_clock::time_point);