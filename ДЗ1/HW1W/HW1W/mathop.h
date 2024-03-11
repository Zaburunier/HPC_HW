#pragma once
#include "stdio.h"
#include "math.h"
#include <immintrin.h> 
#include "omp.h"
//#include "intrin.h"
#include "immintrin.h"


// RU: Внешний метод C = a * (A * B) + b * C
// EN: 
void dgemm(int, int, int, double*, double*, double*&, float = 1.0, float = 0.0);


// RU: Внешний метод C = a * (A * B) + b * C (последовательная версия для проверки)
// EN: 
void dgemm_serial(int, int, int, double*, double*, double*&, float = 1.0, float = 0.0);


// RU: Проверочный метод с последовательным умножением матриц
// EN: 
void blas_dgemm_serial(int, int, int, double*, double*, double*&);


// RU: Метод умножения матриц по заданному в требованиях к ДЗ интерфейсу
// EN: 
void blas_dgemm(int, int, int, double*, double*, double*&);


void dgemm_scal(int, int, int, double *, double *, double *&);


void dgemm_reduction(int, int, int, double *, double *, double *&);


void dgemm_vec(int, int, int, double *, double *, double *&);


void dgemm_avx(int, int, int, double *, double *, double *&);