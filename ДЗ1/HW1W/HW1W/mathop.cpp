#include "mathop.h"
#include "output.h"

void dgemm_vec_initzero(int, int, int, double*);
void dgemm_vec_core(int, int, int, double*, double*, double*);
void dgemm_avx_initzero(int, int, int, double *);
void dgemm_avx_core(int, int, int, double *, double *, double *);


void dgemm(int M, int N, int K, double* A, double* B, double*&C, float alpha, float beta) {
	double* tempC = new double[M * K];
	blas_dgemm(M, N, K, A, B, tempC);
	if (alpha == 1.0 && beta == 0.0) return;

	printf("\nRU/EN - Промежуточный результат (после умножения матриц) // Intermediate result (A * B):\n");
	PrintMatrix(tempC, M, K);

	int length = M * K, i;
	#pragma omp parallel for private(i) shared(tempC, C)
	for (i = 0; i < length; i = i + 1) {
		C[i] = alpha * tempC[i] + beta * C[i];
	}
}


void dgemm_serial(int M, int N, int K, double* A, double* B, double*&C, float alpha, float beta) {
	double* tempC = new double[M * K];
	blas_dgemm_serial(M, N, K, A, B, tempC);
	if (alpha == 1.0 && beta == 0.0) return;

	printf("\nRU/EN - Промежуточный результат (после умножения матриц) // Intermediate result (A * B):\n");
	PrintMatrix(tempC, M, K);

	int length = M * K, i;
	for (i = 0; i < length; i = i + 1) {
		C[i] = alpha * tempC[i] + beta * C[i];
	}
}


void blas_dgemm_serial(int M, int N, int K, double *A, double *B, double *&C)
{
	int i, j, k;
	
	for (i = 0; i < K; i = i + 1) {
		int bColumnBase = i * N, cColumnBase = i * M;

		for (j = 0; j < M; j = j + 1) {
			int aColumnBase = j * M;
			double bValue = B[bColumnBase + j];

			for (int k = 0; k < N; k = k + 1) {
				C[j + i * M] += A[j + k * M] * B[k + i * N];
			}
		}
	}
}


void blas_dgemm_serial_deprecated(int M, int N, int K, double *A, double *B, double *&C)
{
	int i, j, k;
	
	for (i = 0; i < K; i = i + 1) {
		int bColumnBase = i * N, cColumnBase = i * M;

		for (k = 0; k < M; k = k + 1)
			C[cColumnBase + k] = 0.0;

		for (j = 0; j < N; j = j + 1) {
			int aColumnBase = j * M;
			double bValue = B[bColumnBase + j];

			for (int k = 0; k < M; k = k + 1)
				C[cColumnBase + k] += A[aColumnBase + k] * bValue;

		}
	}

}



void blas_dgemm(int M, int N, int K, double *A, double *B, double *&C)
{
	//dgemm_scal(M, N, K, A, B, C);
	//dgemm_reduction(M, N, K, A, B, C);
	dgemm_vec(M, N, K, A, B, C);
	//dgemm_avx(M, N, K, A, B, C);
}


void dgemm_scal(int M, int N, int K, double *A, double *B, double *&C)
{
	printf("RU/EN - Умножение матриц с использованием OpenMP... // Calculating matrix product using OpenMP... (parallel for)\n");

	int i, j, k;
	#pragma omp parallel for private(i, j, k) shared(A, B, C)
	for (i = 0; i < K; i = i + 1) {
		int bColumnBase = i * N, cColumnBase = i * M;

		for (k = 0; k < M; k = k + 1) 
			C[cColumnBase + k] = 0.0;

		for (j = 0; j < N; j = j + 1) {
			int aColumnBase = j * M;
			double bValue = B[bColumnBase + j];

			for (int k = 0; k < M; k = k + 1) 
				C[cColumnBase + k] += A[aColumnBase + k] * bValue;
		}
	}
	
}


void dgemm_reduction(int M, int N, int K, double *A, double *B, double *&C)
{
	printf("RU/EN - Умножение матриц с использованием OpenMP... // Calculating matrix product using OpenMP... (parallel for + reduction)\n");

	int i, j, k;
	double cValue;
	#pragma omp parallel
	{
		for (int i = 0; i < K; i = i + 1) {
			for (int j = 0; j < M; j = j + 1) {
				cValue = 0.0;
				//C[j + i * M] = 0.0;
				#pragma omp for reduction(+:cValue)
				for (int k = 0; k < N; k = k + 1) {
					//printf("C[%d, %d] += A[%d, %d] * B[%d, %d]\n", j, i, j, k, k, i);
					cValue += A[j + k * M] * B[k + i * N];
				}

				C[j + i * M] = cValue;
			}
		}
	}


}


void dgemm_vec(int M, int N, int K, double *A, double *B, double *&C)
{
	printf("RU/EN - Умножение матриц с использованием OpenMP и SSE4.2... // Calculating matrix product using OpenMP and SSE4.2...\n");

	int i, j, k;
	int	outerLength = M / 8,
		outerResidual = M % 8,
		innerLength = K / 2,
		innerResidual = K % 2;
	#pragma omp parallel for private(i, j, k)// shared(A, B, C)
	for (i = 0; i < outerLength; i = i + 1) {
		int iBaseIndex = 8 * i;
		//printf("i = %d\n", iBaseIndex);

		// RU: основную часть матрицы считаем с помощью векторных операций
		// EN: calculating main parts with vectorization
		for (j = 0; j < innerLength; j = j + 1) {
			int jBaseIndex = 2 * j;
			//printf("i = %d, j = %d\n", iBaseIndex, jBaseIndex);
			dgemm_vec_initzero(8, 2, M, C + iBaseIndex + jBaseIndex * M);
			dgemm_vec_core(M, N, K, A + iBaseIndex, B + jBaseIndex * N, C + iBaseIndex + jBaseIndex * M);

			/*printf("After i = %d, j = %d\n", iBaseIndex, jBaseIndex);
			PrintMatrix(C, M, K);
			printf("\n");*/
		}

		// RU: края матрицы считаем "в лоб"
		// EN: calculating edges of matrix with naive implementation
		if (innerResidual > 0) {
			for (j = 0; j < 8; j = j + 1) {
				int jBaseIndex = 2 * innerLength;
				float value = 0;
				for (k = 0; k < N; k = k + 1) {
					value += A[iBaseIndex + j + k * M] * B[k + jBaseIndex * N];
				}

				C[iBaseIndex + j + jBaseIndex * M] = value;
			}
		}

		/*printf("After i = %d", iBaseIndex);
		PrintMatrix(C, M, K);
		printf("\n");*/
	}

	// RU: края матрицы считаем "в лоб"
	// EN: calculating edges of matrix with naive implementation
	if (outerResidual > 0) {
		for (i = 0; i < K; i = i + 1) {
			int bColumnBase = i * N, cColumnBase = i * M;

			for (k = 8 * outerLength; k < M; k = k + 1)
				C[cColumnBase + k] = 0.0;

			for (j = 0; j < N; j = j + 1) {
				int aColumnBase = j * M;
				double bValue = B[bColumnBase + j];

				//int offset = ;
				for (k = 8 * outerLength; k < M; k = k + 1)
					C[cColumnBase + k] += A[aColumnBase + k] * bValue;
			}
		}
	}
}


void dgemm_vec_initzero(int rows, int columns, int step, double *C) {
	for (int i = 0; i < columns; i = i + 1) {
		for (int j = 0; j < rows; j = j + 2)
			_mm_storeu_pd(C + j, _mm_setzero_pd());

		C += step;
	}
}

void dgemm_vec_core(int M, int N, int K, double *A, double *B, double *C)
{
	__m128d c0 = _mm_setzero_pd(),
		c1 = _mm_setzero_pd(),
		c2 = _mm_setzero_pd(),
		c3 = _mm_setzero_pd(),
		c4 = _mm_setzero_pd(),
		c5 = _mm_setzero_pd(),
		c6 = _mm_setzero_pd(),
		c7 = _mm_setzero_pd();

	__m128d a0, a1;
	__m128d b0, b1;

	int	bColumn0 = 0,
		bColumn1 = N;

	// RU: идея в том, что мы храним кусок матрицы C в XMM-регистрах (для избежания постоянных обращений к памяти)
	// EN: main idea is to store matrix C in XMM registers instead of main memory
	for (int i = 0; i < N; i = i + 1) {
		a0 = _mm_loadu_pd(A);
		a1 = _mm_loadu_pd(A + 2);
		b0 = _mm_set1_pd(B[bColumn0]);
		b1 = _mm_set1_pd(B[bColumn1]);

		__m128d temp = _mm_mul_pd(a0, b0);
		c0 = _mm_add_pd(c0, temp);
		temp = _mm_mul_pd(a1, b0);
		c1 = _mm_add_pd(c1, temp);
		temp = _mm_mul_pd(a0, b1);
		c2 = _mm_add_pd(c2, temp);
		temp = _mm_mul_pd(a1, b1);
		c3 = _mm_add_pd(c3, temp);

		a0 = _mm_loadu_pd(A + 4);
		a1 = _mm_loadu_pd(A + 6);

		temp = _mm_mul_pd(a0, b0);
		c4 = _mm_add_pd(c4, temp);
		temp = _mm_mul_pd(a1, b0);
		c5 = _mm_add_pd(c5, temp);
		temp = _mm_mul_pd(a0, b1);
		c6 = _mm_add_pd(c6, temp);
		temp = _mm_mul_pd(a1, b1);
		c7 = _mm_add_pd(c7, temp);

		A += M;
		B += 1;
	}

	_mm_storeu_pd(C + 0, _mm_add_pd(c0, _mm_loadu_pd(C + 0)));
	_mm_storeu_pd(C + 2, _mm_add_pd(c1, _mm_loadu_pd(C + 2)));
	_mm_storeu_pd(C + 4, _mm_add_pd(c4, _mm_loadu_pd(C + 4)));
	_mm_storeu_pd(C + 6, _mm_add_pd(c5, _mm_loadu_pd(C + 6)));

	C += M;
	_mm_storeu_pd(C + 0, _mm_add_pd(c2, _mm_loadu_pd(C + 0)));
	_mm_storeu_pd(C + 2, _mm_add_pd(c3, _mm_loadu_pd(C + 2)));
	_mm_storeu_pd(C + 4, _mm_add_pd(c6, _mm_loadu_pd(C + 4)));
	_mm_storeu_pd(C + 6, _mm_add_pd(c7, _mm_loadu_pd(C + 6)));
}


void dgemm_avx(int M, int N, int K, double *A, double *B, double *&C)
{
	printf("RU/EN - Умножение матриц с использованием OpenMP и AVX512... // Calculating matrix product using OpenMP and AVX512...\n");

	int i, j, k;
	int	outerLength = M / 48,
		outerResidual = M % 48,
		innerLength = K / 2,
		innerResidual = K % 2;
	#pragma omp parallel for private(i, j, k) shared(A, B, C)
	for (i = 0; i < outerLength; i = i + 1) {
		int iBaseIndex = 48 * i;

		// RU: основную часть матрицы считаем с помощью векторных операций
		// EN: calculating main parts with vectorization
		for (j = 0; j < innerLength; j = j + 1) {
			int jBaseIndex = 2 * j;
			dgemm_avx_initzero(48, 2, M, C + iBaseIndex + jBaseIndex * M);
			dgemm_avx_core(M, N, K, A + iBaseIndex, B + jBaseIndex * N, C + iBaseIndex + jBaseIndex * M);

			//printf("After i = %d, j = %d\n", iBaseIndex, jBaseIndex);
			//PrintMatrix(C, M, K);
			//printf("\n");
		}

		// RU: края матрицы считаем "в лоб"
		// EN: calculating edges of matrix with naive implementation
		if (innerResidual > 0) {
			for (j = 0; j < 48; j = j + 1) {
				int jBaseIndex = 2 * innerLength;
				float value = 0;
				for (k = 0; k < N; k = k + 1) {
					value += A[iBaseIndex + j + k * M] * B[k + jBaseIndex * N];
				}

				C[iBaseIndex + j + jBaseIndex * M] = value;
			}
		}

		//printf("After i = %d", iBaseIndex);
		//PrintMatrix(C, M, K);
		//printf("\n");
	}

	// RU: края матрицы считаем "в лоб"
	// EN: calculating edges of matrix with naive implementation
	if (outerResidual > 0) {
		for (i = 0; i < K; i = i + 1) {
			int bColumnBase = i * N, cColumnBase = i * M;

			for (k = 48 * outerLength; k < M; k = k + 1)
				C[cColumnBase + k] = 0.0;

			for (j = 0; j < N; j = j + 1) {
				int aColumnBase = j * M;
				double bValue = B[bColumnBase + j];

				//int offset = ;
				for (k = 48 * outerLength; k < M; k = k + 1)
					C[cColumnBase + k] += A[aColumnBase + k] * bValue;
			}
		}
	}

}


void dgemm_avx_initzero(int rows, int columns, int step, double *C) {
	for (int i = 0; i < columns; i = i + 1) {
		for (int j = 0; j < rows; j = j + 8)
			_mm512_storeu_pd(C + j, _mm512_setzero_pd());

		C += step;
	}

}


void dgemm_avx_core(int M, int N, int K, double *A, double *B, double *C)
{
	__m512d c0 = _mm512_setzero_pd(),
		c1 = _mm512_setzero_pd(),
		c2 = _mm512_setzero_pd(),
		c3 = _mm512_setzero_pd(),
		c4 = _mm512_setzero_pd(),
		c5 = _mm512_setzero_pd(),
		c6 = _mm512_setzero_pd(),
		c7 = _mm512_setzero_pd(),
		c8 = _mm512_setzero_pd(),
		c9 = _mm512_setzero_pd(),
		c10 = _mm512_setzero_pd(),
		c11 = _mm512_setzero_pd();

	int	bColumn0 = 0,
		bColumn1 = N;

	// RU: идея в том, что мы храним кусок матрицы C в XMM-регистрах (для избежания постоянных обращений к памяти)
	// EN: main idea is to store matrix C in XMM registers instead of main memory
	for (int i = 0; i < N; i = i + 1) {
		__m512d a0, a1;
		__m512d b0, b1;

		a0 = _mm512_loadu_pd(A);
		a1 = _mm512_loadu_pd(A + 8);
		b0 = _mm512_set1_pd(B[bColumn0]);
		b1 = _mm512_set1_pd(B[bColumn1]);

		c0 = _mm512_fmadd_pd(a0, b0, c0);
		c1 = _mm512_fmadd_pd(a1, b0, c1);
		c2 = _mm512_fmadd_pd(a0, b1, c2);
		c3 = _mm512_fmadd_pd(a1, b1, c3);

		a0 = _mm512_loadu_pd(A + 16);
		a1 = _mm512_loadu_pd(A + 24);

		c4 = _mm512_fmadd_pd(a0, b0, c4);
		c5 = _mm512_fmadd_pd(a1, b0, c5);
		c6 = _mm512_fmadd_pd(a0, b1, c6);
		c7 = _mm512_fmadd_pd(a1, b1, c7);

		a0 = _mm512_loadu_pd(A + 32);
		a1 = _mm512_loadu_pd(A + 40);

		c8 = _mm512_fmadd_pd(a0, b0, c8);
		c9 = _mm512_fmadd_pd(a1, b0, c9);
		c10 = _mm512_fmadd_pd(a0, b1, c10);
		c11 = _mm512_fmadd_pd(a1, b1, c11);

		A += M;
		B += 1;
	}

	_mm512_storeu_pd(C + 0, _mm512_add_pd(c0, _mm512_loadu_pd(C + 0)));
	_mm512_storeu_pd(C + 8, _mm512_add_pd(c1, _mm512_loadu_pd(C + 8)));
	_mm512_storeu_pd(C + 16, _mm512_add_pd(c4, _mm512_loadu_pd(C + 16)));
	_mm512_storeu_pd(C + 24, _mm512_add_pd(c5, _mm512_loadu_pd(C + 24)));
	_mm512_storeu_pd(C + 32, _mm512_add_pd(c8, _mm512_loadu_pd(C + 32)));
	_mm512_storeu_pd(C + 40, _mm512_add_pd(c9, _mm512_loadu_pd(C + 40)));

	C += M;
	_mm512_storeu_pd(C + 0, _mm512_add_pd(c2, _mm512_loadu_pd(C + 0)));
	_mm512_storeu_pd(C + 8, _mm512_add_pd(c3, _mm512_loadu_pd(C + 8)));
	_mm512_storeu_pd(C + 16, _mm512_add_pd(c6, _mm512_loadu_pd(C + 16)));
	_mm512_storeu_pd(C + 24, _mm512_add_pd(c7, _mm512_loadu_pd(C + 24)));
	_mm512_storeu_pd(C + 32, _mm512_add_pd(c10, _mm512_loadu_pd(C + 32)));
	_mm512_storeu_pd(C + 40, _mm512_add_pd(c11, _mm512_loadu_pd(C + 40)));
}