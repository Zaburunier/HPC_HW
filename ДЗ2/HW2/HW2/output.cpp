#include "output.h"

#include <fstream>
#include <string>


bool AreEqual(double* first, double* second, int length, float threshold) {
	int i;
	bool result = true;

	for (i = 0; i < length; i = i + 1) {
		if (abs(first[i] - second[i]) >= threshold) return false;
	}

	return true;
}


// RU: Вывод матрицы на консоль в удобном виде
// EN: Nice-formatted matrix output
void PrintMatrix(double *matrix, int rows, int columns)
{
	if (matrix == nullptr) {
		printf("Got nullptr, printing is impossible.\n\n");
		return;
	}

	if (rows > 64 || columns > 64) {
		printf("Output is not supported for matrices larger than 64 (any dim)\n\n");
		return;
	}

	printf("----------------------\n");
	for (int i = 0; i < rows; i = i + 1) {
		printf("(");
		//PrintArray(matrix + i * columns, columns);
		for (int j = 0; j < columns - 1; j = j + 1) {
			printf("%.2f\t", matrix[j * rows + i]);
		}
		printf("%.2f)\n", matrix[(columns - 1) * rows + i]);
	}
	printf("----------------------\n");
}


void PrintArray(double *array, int length)
{
	for (int j = 0; j < length - 1; j = j + 1) {
		printf("%.2f ", array[j]);
	}

	printf("%.2f", array[length - 1]);
}


void WriteMatrixToFile(double *matrix, char* path, int &rows, int &columns)
{
	std::ofstream file;

	file.open(path, std::ios::out);
	if (file.is_open() == false) {
		return;
		/*file.
		if (file.is_open()) {
			printf("RU - Невозможно открыть файл с матрицей, возврат единичной матрицы 4x4.\n");
			printf("EN - Can`t open input matrix file, returning 4x4 identity matrix.\n\n");
			return;
		}*/
	}

	int	outerLength = rows - 1,
		innerLength = columns - 1;
	for (int i = 0; i < outerLength; i = i + 1) {
		for (int j = 0; j < innerLength; j = j + 1) {
			file.write(std::to_string(matrix[rows * i + j]).c_str(), sizeof(double));
		}

		file.write(std::to_string(matrix[rows * (i + 1) - 1]).c_str(), sizeof(double));
		file.write("\n", 2);
	}

	for (int j = 0; j < innerLength; j = j + 1) {
		file.write(std::to_string(matrix[rows * (rows - 1) + j]).append(" ").c_str(), sizeof(double) + 1);
	}

	file.write(std::to_string(matrix[rows * rows - 1]).c_str(), sizeof(double));
	file.write("\n", 2);

	file.close();
}


double ConvertChronoToSeconds(std::chrono::system_clock::time_point beginTimeStamp, std::chrono::system_clock::time_point endTimeStamp) {
	return 1.0e-06 * std::chrono::duration_cast<std::chrono::microseconds>(endTimeStamp - beginTimeStamp).count();
}

