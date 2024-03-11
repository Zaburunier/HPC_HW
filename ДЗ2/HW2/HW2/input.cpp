#define _CRT_SECURE_NO_WARNINGS


#include "input.h"
#include "random"
#include <string>


// RU: Смена представления матрицы, заданной массивом, с row-major order на column-major order
// EN: Converting matrix from row-major order to column-major order
double* RowMajorToColumnMajor(double *matrix, int rows, int columns)
{
	// RU: Row-major означает запись элементов одной строки подряд, то есть A[i,j] => matrix[i * columns + j]
	// Column-major изменит этот порядок на A[i,j] => matrix[j * rows + i] (подряд записаны элементы столбца)
	// EN: Row-major order means A[i,j] => matrix[i * columns + j] (row elements are next to each other)
	// Column major order meands A[i,j] => matrix[j * rows + i] (column elements are next to each other)

	int length = rows * columns;
	double *newMatrix = new double[length];

	for (int i = 0; i < length; i = i + 1) {
		int column = i / columns, row = i % columns;
		newMatrix[column * columns + row] = matrix[i];
	}

	return newMatrix;
}


double* ParseLine(std::string line, int& numCounter)
{
	numCounter = 0;

	double* numbers = new double[0];
	int length = line.length();
	std::string numAsString;

	for (int i = 0; i < length; i = i + 1) {
		if (line[i] != ' ') {
			numAsString += line[i];
		} else {
			// RU: завершили считывание числа, обновляем массив
			// EN: found separator, appending to numbers array
			numCounter += 1;
			double* newNumbers = new double[numCounter];

			for (int j = 0; j < numCounter; j = j + 1) {
				newNumbers[j] = numbers[j];
			}

			newNumbers[numCounter - 1] = std::stod(numAsString);
			numbers = newNumbers;
			numAsString = "";
		}
	}

	// RU: последний элемент идёт без разделителя
	// EN: last element has no separator
	numCounter += 1;
	double* newNumbers = new double[numCounter];

	for (int j = 0; j < numCounter; j = j + 1) {
		newNumbers[j] = numbers[j];
	}

	newNumbers[numCounter - 1] = std::stod(numAsString);
	numbers = newNumbers;
	numAsString = "";

	return numbers;
}


// RU: Чтение файла с матрицей
// EN: Reading file to get matrix
double* GetMatrixFromFile(char *path, int& rows, int& columns, bool toCudaPinnedMemory)
{
	std::ifstream file;

	file.open(path, std::ios::in);
	if (file.is_open() == false) {
		printf("RU - Невозможно открыть файл с матрицей, возврат единичной матрицы 4x4.\n");
		printf("EN - Can`t open input matrix file, returning 4x4 identity matrix.\n\n");

		return GetIdentityMatrix(4);
	}

	double *fileMatrix;
	if (toCudaPinnedMemory == false) {
		fileMatrix = new double[0];
	} else {
		cudaHostAlloc(&fileMatrix, 0, cudaHostAllocDefault);
	}

	std::string fileLine;
	rows = 0;
	columns = 0;

	while (std::getline(file, fileLine)) {
		rows += 1;

		int length;
		double* numbers = ParseLine(fileLine, length);
		if (columns != 0 && length != columns) {
			printf("RU - Ошибка при чтении файла с матрицей: различное кол-во чисел в строке.\n");
			printf("EN - Error while reading input matrix file: line length mismatch.\n\n");

			return GetIdentityMatrix(4);
		}

		columns = length;

		double* newMatrix;
		if (toCudaPinnedMemory == false) {
			newMatrix = new double[rows * columns];
		} else {
			cudaHostAlloc(&newMatrix, rows * columns * sizeof(double), cudaHostAllocDefault);
		}

		int arrayLength = (rows - 1) * columns;

		for (int i = 0; i < arrayLength; i = i + 1) {
			newMatrix[i] = fileMatrix[i];
		}

		for (int i = 0; i < columns; i = i + 1) {
			newMatrix[arrayLength + i] = numbers[i];
		}

		fileMatrix = newMatrix;
	}

	file.close();
	return fileMatrix;
}


// RU: Получение матрицы от пользователя
// EN: Reading user input to get matrix
double* GetMatrixFromConsoleInput(int& rows, int& columns, bool toCudaPinnedMemory)
{
	printf("RU/EN - Введите кол-во строк // Set rows amounts: ");
	scanf("%d", &rows);

	printf("RU/EN - Введите кол-во столбцов // Set columns amounts: ");
	scanf("%d", &columns);

	printf("RU - Введите матрицу, разделяя строки переносом, а элементы одной строки - пробелом\n");
	printf("EN - Enter matrix values (row separator - CR, column separator - space)\n");

	double* inputMatrix;
	if (toCudaPinnedMemory == false) {
		inputMatrix = new double[rows * columns];
	} else {
		cudaHostAlloc(&inputMatrix, rows * columns * sizeof(double), cudaHostAllocDefault);
	}

	for (int i = 0; i < rows; i = i + 1) {
		char dummy;

		for (int j = 0; j < columns - 1; j = j + 1){
			scanf("%lf ", &(inputMatrix[i * columns + j]));
		}

		scanf("%lf", &(inputMatrix[(i + 1) * columns - 1]));
		scanf("%c", &dummy);
	}
	printf("\n");

	return inputMatrix;
}


// RU: Создание матрицы генератором случайных чисел
// EN: Randomly generating matrix
double* GetRandomMatrix(int& rows, int& columns, double minInclusive, double maxExclusive, bool toCudaPinnedMemory)
{
	printf("RU/EN - Введите кол-во строк // Set rows amounts: ");
	scanf("%d", &rows);

	printf("RU/EN - Введите кол-во столбцов // Set columns amounts: ");
	scanf("%d", &columns);

	int length = rows * columns;
	double intervalLength = maxExclusive - minInclusive,
	*result;

	if (toCudaPinnedMemory == false) {
		result = new double[length];
	} else {
		cudaHostAlloc(&result, length * sizeof(double), cudaHostAllocDefault);
	}

	for (int i = 0; i < length; i = i + 1) {
		result[i] = minInclusive + rand() * intervalLength / RAND_MAX;
	}

	return result;
}


// RU: Создание едичной матрицы
// EN: Generating identity matrix
double* GetIdentityMatrix(int dims, bool toCudaPinnedMemory)
{
	int length = dims * dims;
	double *result;
	if (toCudaPinnedMemory == false) {
		result = new double[length];
	} else {
		cudaHostAlloc(&result, length * sizeof(double), cudaHostAllocDefault);
	}

	for (int i = 0; i < dims; i = i + 1) {
		for (int j = 0; j < i; j = j + 1)
			result[i * dims + j] = 0.0;

		result[i * dims + i] = 1.0;

		for (int j = i + 1; j < dims; j = j + 1)
			result[i * dims + j] = 0.0;
	}

	return result;
}

