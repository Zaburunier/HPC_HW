#include <chrono>

#include "stdio.h"
#include "input.h"
#include "kernels.cuh"
#include "output.h"
#include "mathop.h"

constexpr auto USE_CUDA_PINNED_MEMORY = true;


int main(int argc, char* argv[])
{
	printf("MSMT221, Zaburunov Leonid V., High-Performance Computations, Homework #2.\n\n");

	if (USE_CUDA_PINNED_MEMORY) {
		printf("Pinned memory allocation enabled...\n\n");
	}

	double *A = nullptr, *B = nullptr, *C = nullptr;
	int aRows = 0, aColumns = 0, bRows = 0, bColumns = 0, cRows = 0, cColumns = 0;

	printf("\n----------------\n");
	printf("ANALYZING INPUT");
	printf("\n--------------------------------\n");

	if (argc > 1 && argv[1] == std::string("random")) {
		double minValue = argc < 3 ? 0.0 : std::atof(argv[2]),
			maxValue = argc < 4 ? 0.0 : std::atof(argv[3]);

		printf("Initializing matrices randomly from [%e; %e)...\n\n", minValue, maxValue);

		printf("First matrix.\n");
		A = GetRandomMatrix(aRows, aColumns, minValue, maxValue, USE_CUDA_PINNED_MEMORY);

		printf("\nSecond matrix.\n");
		B = GetRandomMatrix(bRows, bColumns, minValue, maxValue, USE_CUDA_PINNED_MEMORY);

		cRows = aRows;
		cColumns = bColumns;
	} else if (argc > 1 && argv[1] == std::string("identity")) {
		int dim = argc < 3 ? 10 : std::atoi(argv[2]);
		printf("Initializing identity square matrices with dimensions %d...\n\n", dim);

		A = GetIdentityMatrix(dim, USE_CUDA_PINNED_MEMORY);
		B = GetIdentityMatrix(dim, USE_CUDA_PINNED_MEMORY);

		aRows = dim;
		aColumns = dim;
		bRows = dim;
		bColumns = dim;
		cRows = aRows;
		cColumns = bColumns;
	}

	if (aColumns != bRows || cRows != aRows || cColumns != bColumns) {
		printf("ERROR! Matrix dimensions mismatch ([%d, %d] <--X--> [%d, %d] <--X--> [%d, %d]). Product computation cancelled.\n\n", aRows, aColumns, bRows, bColumns, cRows, cColumns);
		return 1;
	}


	if (USE_CUDA_PINNED_MEMORY == false) {
		C = static_cast<double*>(calloc(cRows * cColumns, sizeof(double)));
	} else {
		cudaHostAlloc(&C, cRows * cColumns * sizeof(double), cudaHostAllocDefault);
		for (int i = 0; i < cColumns; i = i + 1) {
			for (int j = 0; j < cRows; j = j + 1) {
				C[i * cRows + j] = 0.0;
			}
		}
	}

	printf("First matrix:\n");
	/*PrintArray(A, aRows * aColumns);
	printf("\n(column-major order)\n");*/
	PrintMatrix(A, aRows, aColumns);

	printf("\Second matrix:\n");
	/*PrintArray(B, bRows * bColumns);
	printf("\n(column-major order)\n");*/
	PrintMatrix(B, bRows, bColumns);

	printf("\n--------------------------------\n\n");
	printf("COMPUTATIONAL PART");
	printf("\n--------------------------------\n\n");

	auto t0 = std::chrono::system_clock::now();
	cuda_mm(aRows, aColumns, bColumns, A, B, C);
	auto t = std::chrono::system_clock::now();
	auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(t - t0).count();

	printf("\nResult:\n");
	PrintMatrix(C, aRows, bColumns);
	printf("Op time complexity: %f s.\n", ConvertChronoToSeconds(t0 ,t));

	//return 0;

	printf("\n--------------------------------\n");
	printf("VERIFIYING PART");
	printf("\n--------------------------------\n");

	double *serialC = static_cast<double*>(calloc(cRows * cColumns, sizeof(double)));

	t0 = std::chrono::system_clock::now();
	blas_dgemm_serial(aRows, aColumns, bColumns, A, B, serialC);
	t = std::chrono::system_clock::now();
	elapsed = std::chrono::duration_cast<std::chrono::microseconds>(t - t0).count();

	printf("Result with serial CPU implementation:\n");
	PrintMatrix(serialC, aRows, bColumns);
	printf("Op time complexity: %f s.\n", ConvertChronoToSeconds(t0, t));
	printf("Parallel implementation is correct: %s", AreEqual(C, serialC, cRows * cColumns) ? "True\n" : "False\n");

	return 0;
}
