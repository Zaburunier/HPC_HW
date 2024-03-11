// HW3.cpp : Этот файл содержит функцию "main". Здесь начинается и заканчивается выполнение программы.
//

#include <ostream>

#include "constants.h"
#include "conductivity_solver.h"
#include "stdio.h"
#include "mpi.h"
#include <thread>


int main(int argc, char **argv) {
	double mpiTime = 0.0, seriesTime = 0.0;
	int processRank, amountOfProcessors;
	double* pointsInfo = new double[2];

	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &amountOfProcessors);
	MPI_Comm_rank(MPI_COMM_WORLD, &processRank);

	double *processorPointsInfo = new double[amountOfProcessors * 2];
	int *gatherCounts = new int[amountOfProcessors], *gatherBaseIndices = new int[amountOfProcessors];

	// Для каждого процессора формируем по два числа: левую и правую границу
	if (processRank == 0) {
		printf("MSMT221, Zaburunov Leonid V., High-Performance Computations, Homework #3.\n\n");
		printf("Using %d processes for calculation.\nSetup: rod length = %f; target time is %e; total points = %d; equation ratio: %e.\n\n", amountOfProcessors, ROD_LENGTH, TIME_LENGTH, N_POINTS, EQUATION_RATIO);
		
		int	div = N_POINTS / amountOfProcessors,
			mod = N_POINTS % amountOfProcessors;

		int processPoints = div + (mod > 0 ? 0 : -1);
		processorPointsInfo[0] = 0;
		processorPointsInfo[1] = SPATIAL_STEP * processPoints;
		gatherCounts[0] = processPoints + 1;
		gatherBaseIndices[0] = 0;

		for (int i = 1; i < amountOfProcessors; i = i + 1) {
			processPoints = div + (i <= mod - 1 ? 0 : -1);
			processorPointsInfo[i * 2] = processorPointsInfo[(i - 1) * 2 + 1] + SPATIAL_STEP;
			processorPointsInfo[i * 2 + 1] = processorPointsInfo[i * 2] + SPATIAL_STEP * processPoints;
			gatherCounts[i] = processPoints + 1;

			gatherBaseIndices[i] = 0;
			for (int j = 0; j < i; j = j + 1) {
				gatherBaseIndices[i] += gatherCounts[j];
			}
		}

		for (int i = 0; i < amountOfProcessors; i = i + 1) {
			printf("Calculating points [%f; %f] with step %f (%d points starting with index %d) on processor %d\n", processorPointsInfo[i * 2], processorPointsInfo[i * 2 + 1], SPATIAL_STEP, gatherCounts[i], gatherBaseIndices[i], i + 1);
		}

		//std::flush(stdout);
	}

	// Коллективная рассылка
	MPI_Scatter(processorPointsInfo, 2, MPI_DOUBLE, pointsInfo, 2, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Barrier(MPI_COMM_WORLD);
	//delete[] processorPointsInfo;

	double	*pointsData = new double[N_POINTS],
		*deviations = new double[amountOfProcessors],
		*deviationPoints = new double[amountOfProcessors];

	if (processRank == 0) mpiTime = MPI_Wtime();
	SolveRodPiece(pointsInfo[0], pointsInfo[1], static_cast<int>(round((pointsInfo[1] - pointsInfo[0]) / SPATIAL_STEP)) + 1, pointsData, deviations, deviationPoints, gatherCounts, gatherBaseIndices, processRank, amountOfProcessors);
	//delete[] pointsInfo;

	/*MPI_Finalize();
	return 0;*/

	MPI_Barrier(MPI_COMM_WORLD);

	if (processRank == 0) {
		mpiTime = MPI_Wtime() - mpiTime;

		/*double *pointsData = nullptr, *deviations = new double[amountOfProcessors], *deviationPoints = new double[amountOfProcessors];
	int totalLength = 0;

	for (int i = 0; i < amountOfProcessors; i = i + 1) {
		// Принимаем от каждого процесса распределения температуры
		MPI_Status status;
		MPI_Probe(i, 1, MPI_COMM_WORLD, &status);
		int count;
		MPI_Get_count(&status, MPI_DOUBLE, &count);

		double* procResult = new double[count];//static_cast<double*>(calloc(count, sizeof(double)));
		MPI_Recv(procResult, count, MPI_DOUBLE, i, 1, MPI_COMM_WORLD, &status);

		double* newResults = new double[totalLength + count];//static_cast<double*>(calloc(totalLength + count, sizeof(double)));
		for (int i = 0; i < totalLength; i = i + 1) {
			newResults[i] = pointsData[i];
		}
		delete[] pointsData;//free(mpiValues);

		for (int i = 0; i < count; i = i + 1) {
			newResults[totalLength + i] = procResult[i];
		}
		delete[] procResult;//free(procResult);

		pointsData = newResults;
		totalLength = totalLength + count;

		// Принимаем также информацию об отклонениях
		MPI_Recv(deviations + i, 1, MPI_DOUBLE, i, 2, MPI_COMM_WORLD, &status);
		MPI_Recv(deviationPoints + i, 1, MPI_DOUBLE, i, 3, MPI_COMM_WORLD, &status);
	}*/

		printf("\n-----------------------------\n\n");

		for (int i = 0; i < N_POINTS; i = i + 1) {
			printf("Point %f has temp %e\n", i * SPATIAL_STEP, pointsData[i]);
		}

		printf("\n-----------------------------\n\n");

		double totalDeviation = -1.0, totalDeviationPoint = -1.0;
		for (int i = 0; i < amountOfProcessors; i = i + 1) {
			printf("Process %d max error: %e (at point %f)\n", i + 1, deviations[i], deviationPoints[i]);
			if (deviations[i] > totalDeviation) {
				totalDeviation = deviations[i];
				totalDeviationPoint = deviationPoints[i];
			}
		}

		printf("\n-----------------------------\n\n");
		printf("MPI version max error: %e (at point %f)\n", totalDeviation, totalDeviationPoint);

		printf("\n-----------------------------\n\n");
		printf("MPI version time complexity: %e\n", mpiTime);
	}

	MPI_Barrier(MPI_COMM_WORLD);
	//delete[] pointsData;
	//delete[] deviations;
	//delete[] deviationPoints;

	MPI_Finalize();
	return 0;
}