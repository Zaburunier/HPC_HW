#include <cstdio>
#include "omp.h"
#include "input.h"
#include "mathop.h"
#include "output.h"


int main(int argc, char* argv[])
{
	printf("RU - МСМТ221, Забурунов Леонид Вячеславович, МСМТ221, Высокопроизводительные Вычисления, ДЗ #1.\n");
	printf("EN - MSMT221, Zaburunov Leonid V., High-Performance Computations, Homework #1.\n\n");

	double *A = nullptr, *B = nullptr, *C = nullptr;
	int aRows = 0, aColumns = 0, bRows = 0, bColumns = 0, cRows = 0, cColumns = 0;

	bool hasThreeMatricesAsInput = false;

	float alpha = 1.0, beta = 0.0;
	printf("\n----------------\n");
	printf("RU: АНАЛИЗ ВХОДНЫХ ДАННЫХ\n");
	printf("EN: ANALYZING INPUT");
	printf("\n--------------------------------\n");

	if (argc > 1 && argv[1] == std::string("random")) {
		double minValue = argc < 3 ? 0.0 : std::atof(argv[2]),
		maxValue = argc < 4 ? 0.0 : std::atof(argv[3]);

		printf("RU - Проводим случайную инициализацию матриц значениями на интервале [%e; %e)...\n", minValue, maxValue);
		printf("EN - Initializing matrices randomly from [%e; %e)...\n\n", minValue, maxValue);

		printf("RU/EN - Первая матрица // First matrix.\n");
		A = GetRandomMatrix(aRows, aColumns, minValue, maxValue);

		printf("\nRU/EN - Вторая матрица // Second matrix.\n");
		B = GetRandomMatrix(bRows, bColumns, minValue, maxValue);

		cRows = aRows;
		cColumns = bColumns;
	} else if (argc > 1 && argv[1] == std::string("identity")) {
		int dim = argc < 3 ? 10 : std::atoi(argv[2]);
		printf("RU - Проводим инициализацию единичных квадратных матриц с размерностью %d...\n", dim);
		printf("EN - Initializing identity square matrices with dimensions %d...\n\n", dim);

		A = GetIdentityMatrix(dim);
		B = GetIdentityMatrix(dim);

		aRows = dim;
		aColumns = dim;
		bRows = dim;
		bColumns = dim;
		cRows = aRows;
		cColumns = bColumns;
	} else if (argc == 1) {
		printf("RU - Файлы с исходными данными не заданы, обе матрицы будут получены через консоль...\n");
		printf("EN - Input files not set, getting matrices from console input...\n\n");
		
		printf("\nRU/EN - Первая матрица // First matrix.\n");
		A = GetMatrixFromConsoleInput(aRows, aColumns);
		A = RowMajorToColumnMajor(A, aRows, aColumns);

		printf("\nRU/EN - Вторая матрица // Second matrix.\n");
		B = GetMatrixFromConsoleInput(bRows, bColumns);
		B = RowMajorToColumnMajor(B, bRows, bColumns);

		cRows = aRows;
		cColumns = bColumns;
	} else if (argc == 2) {
		printf("RU - Задан файл с исходными данными для первой матрицы...\n");
		printf("EN - Found file with first matrix...\n\n");
		A = GetMatrixFromFile(argv[1], aRows, aColumns);
		A = RowMajorToColumnMajor(A, aRows, aColumns);

		printf("RU - Используем тот же файл для получения второй матрицы...\n");
		printf("EN - Using same file for second matrix...\n\n");
		B = GetMatrixFromFile(argv[1], bRows, bColumns);
		B = RowMajorToColumnMajor(B, bRows, bColumns);

		cRows = aRows;
		cColumns = bColumns;
	} else if (argc == 3) {
		printf("RU - Задан файл с исходными данными для первой матрицы...\n");
		printf("EN - Found file with first matrix...\n\n");
		A = GetMatrixFromFile(argv[1], aRows, aColumns);
		A = RowMajorToColumnMajor(A, aRows, aColumns);

		printf("RU - Задан файл с исходными данными для второй матрицы...\n");
		printf("EN - Found file with second matrix...\n\n");
		B = GetMatrixFromFile(argv[2], bRows, bColumns);
		B = RowMajorToColumnMajor(B, bRows, bColumns);

		cRows = aRows;
		cColumns = bColumns;
	} else if (argc >= 4) {
		printf("RU - Задан файл с исходными данными для первой матрицы...\n");
		printf("EN - Found file with first matrix...\n\n");
		A = GetMatrixFromFile(argv[1], aRows, aColumns);
		A = RowMajorToColumnMajor(A, aRows, aColumns);

		printf("RU - Задан файл с исходными данными для второй матрицы...\n");
		printf("EN - Found file with second matrix...\n\n");
		B = GetMatrixFromFile(argv[2], bRows, bColumns);
		B = RowMajorToColumnMajor(B, bRows, bColumns);

		printf("RU - Задан файл с исходными данными для третьей матрицы...\n");
		printf("EN - Found file with second matrix...\n\n");
		C = GetMatrixFromFile(argv[3], cRows, cColumns);
		C = RowMajorToColumnMajor(C, cRows, cColumns);

		alpha = argc >= 5 ? std::atof(argv[4]) : alpha;
		beta = argc >= 6 ? std::atof(argv[5]) : beta;

		hasThreeMatricesAsInput = true;
	} else return -1;

	if (aColumns != bRows || cRows != aRows || cColumns != bColumns){
		printf("RU - ОШИБКА! Несогласованные размерности матриц ([%d, %d] <--X--> [%d, %d] <--X--> [%d, %d]). Вычисление произведения невозможно.\n", aRows, aColumns, bRows, bColumns, cRows, cColumns);
		printf("EN - ERROR! Matrix dimensions mismatch ([%d, %d] <--X--> [%d, %d] <--X--> [%d, %d]). Product computation cancelled.\n\n", aRows, aColumns, bRows, bColumns, cRows, cColumns);
		return 1;
	}

	/*if (aRows != aColumns || bRows != bColumns) {
		printf("RU - ОШИБКА! В рамках данного задания можно использовать только квадратные матрицы\n");
		printf("EN - ERROR! Square matrices is the only valid case.\n\n");
		return 2;
	}*/


	printf("RU/EN - Первая матрица // First matrix:\n");
	/*PrintArray(A, aRows * aColumns);
	printf("\n(column-major order)\n");*/
	PrintMatrix(A, aRows, aColumns);

	printf("\nRU/EN - Вторая матрица // Second matrix:\n");
	/*PrintArray(B, bRows * bColumns);
	printf("\n(column-major order)\n");*/
	PrintMatrix(B, bRows, bColumns);

	printf("\n--------------------------------\n\n");
	printf("RU: ВЫЧИСЛИТЕЛЬНЫЙ БЛОК\n");
	printf("EN: COMPUTATIONAL PART");
	printf("\n--------------------------------\n\n");

	double t0, t;
	int cLength = cRows * cColumns;
	double *serialC = static_cast<double*>(calloc(cLength, sizeof(double)));

	if (hasThreeMatricesAsInput) {
		// RU: для проверки нужно внести исходные данные матрицы С
		// EN: copying source C data for result verify
		for (int i = 0; i < cLength; i = i + 1) {
			serialC[i] = C[i];
		}

		printf("RU/EN - Третья матрица // Third matrix:\n");
		/*PrintArray(C, cRows * cColumns);
		printf("\n(column-major order)\n");*/
		PrintMatrix(C, cRows, cColumns);

		printf("\nRU/EN - Проводим вычисления... // Performing computations...\n");
		t0 = omp_get_wtime();
		dgemm(aRows, aColumns, bColumns, A, B, C, alpha, beta);
		t = omp_get_wtime();

		//WriteMatrixToFile(C, argc >= 7 ? argv[6] : (char*) "output.txt", aRows, bColumns);
	} else {
		C = new double[cRows * cColumns];// static_cast<double *>(calloc(cRows *cColumns, sizeof(double)));

		printf("RU/EN - Проводим вычисления... // Performing computations...\n");
		t0 = omp_get_wtime();
		blas_dgemm(aRows, aColumns, bColumns, A, B, C);
		t = omp_get_wtime();
	}

	printf("\nRU/EN - Результат умножения // Result:\n");
	PrintMatrix(C, aRows, bColumns);
	printf("RU/EN - Время исполнения операции // Op time complexity: %e.\n", t - t0);

	printf("\n--------------------------------\n");
	printf("RU: ПРОВЕРОЧНЫЙ БЛОК\n");
	printf("EN: VERIFIYING PART");
	printf("\n--------------------------------\n");

	if (hasThreeMatricesAsInput) {
		t0 = omp_get_wtime();
		dgemm_serial(aRows, aColumns, bColumns, A, B, serialC, alpha, beta);
		t = omp_get_wtime();
	} else {
		t0 = omp_get_wtime();
		blas_dgemm_serial(aRows, aColumns, bColumns, A, B, serialC);
		t = omp_get_wtime();
	}

	printf("RU/EN - Результат умножения через последовательную реализацию // Result with serial implementation:\n");
	PrintMatrix(serialC, aRows, bColumns);
	printf("RU/EN - Время исполнения операции // Op time complexity: %e.\n", t - t0);
	printf("RU/EN - Параллельная реализация корректна // Parallel implementation is correct: %s", AreEqual(C, serialC, cRows *cColumns) ? "True\n" : "False\n");

	return 0;
}