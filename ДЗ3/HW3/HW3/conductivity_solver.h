#pragma once
#include "constants.h"
#include "stdio.h"
#include "math.h"
#include "mpi.h"
#include "omp.h"


// Функция подсчёта температуры куска стержня в MPI-реализации
void SolveRodPiece(double, double, int, double*, double*, double*, int*, int*, int, int);


// Функция подсчёта температуры стержня с помощью вычисления суммы ряда
double* CalculateRodWithSeries(double, double, int = 500);