#include "conductivity_solver.h"

#include <algorithm>
#include <cstdlib>


void SolveRodPiece(double startPoint, double endPoint, int spatialSteps, double* pointsData, double* deviations, double* deviationPoints, int* gatherCounts, int* gatherBeginIndices, int processRank, int amountOfProcessors) {
	bool	isInLeftRodEdge = processRank == 0,
		isInRightRodEdge = processRank == amountOfProcessors - 1;

	double* data = new double[2 * spatialSteps];//static_cast<double*>(calloc(2 * spatialSteps, sizeof(double)));

	// Начальные значения инициализируем вручную
	for (int i = 0; i < spatialSteps; i = i + 1) {
		data[i] = ROD_INITIAL_TEMP;
	}

	MPI_Request request;

	// Рассылаем начальные значения (в цикле на них тоже идёт запрос)
	if (!isInLeftRodEdge) {
		MPI_Send(data, 1, MPI_DOUBLE, processRank - 1, MPI_SEND_LEFT_EDGE_DATA_BASE_TAG, MPI_COMM_WORLD);
		//printf("Proc #%d sent data (%e) on time %f at point %f\n", processRank + 1, *data, 0.0f, startPoint);
	}

	if (!isInRightRodEdge) {
		MPI_Send(data + spatialSteps - 1, 1, MPI_DOUBLE, processRank + 1, MPI_SEND_RIGHT_EDGE_DATA_BASE_TAG, MPI_COMM_WORLD);
		//printf("Proc #%d sent data (%e) on time %f at point %f\n", processRank + 1, *(data + spatialSteps - 1), 0.0f, endPoint);
	}

	// Внешний цикл - по времени, внутренний - по пространству
	// Это допускается, поскольку есть прямая зависимость между результатами для одной точки в разное время,
	// но нет прямой зависимости между результатами для соседних точек в одно время
	for (int i = 1; i < TIME_STEPS; i = i + 1) {
		int	currentTimeBaseIndex = i * spatialSteps,
			lastTimeBaseIndex = (i - 1) * spatialSteps;
		
		// j = 0
		double our = data[0], left, right = data[1];
		if (isInLeftRodEdge) {
			left = ENVIROMENT_TEMP;
		} else {
			// Нумеровать сообщения будем следующим образом:
			// 1. 10 - это чтобы освободить первые номера под общение с главным процессом;
			// 2. i * 2 - это смещение, однозначно определяющее значение времени, к которому относятся данные;
			// 3. + 0 - это левая граница интервала, + 1 - правая
			// Наша левая граница - это правая граница процесса с номером на 1 меньше
			// То есть, 10 + (i - 1) * 2 + 1 - это запрос к процессу, считающему отрезок левее нашего, на температуру его правой границы отрезка в предыдущий момент времени
			MPI_Status status;
			MPI_Recv(&left, 1, MPI_DOUBLE, processRank - 1, MPI_SEND_RIGHT_EDGE_DATA_BASE_TAG + (i - 1) * 2, MPI_COMM_WORLD, &status);
			//if (processRank == 2 && i % 100 == 0) printf("Proc #%d received data (%e) on time %f at point %f\n", processRank + 1, left, i * TIME_STEP, startPoint);
		}

		data[spatialSteps] = our + EQUATION_RATIO * (left - 2 * our + right);
		if (!isInLeftRodEdge) {
			MPI_Send(data + spatialSteps, 1, MPI_DOUBLE, processRank - 1, MPI_SEND_LEFT_EDGE_DATA_BASE_TAG + i * 2, MPI_COMM_WORLD);
			//printf("Proc #%d sent data (%e) on time %f at point %f\n", processRank + 1, *(data), i * TIME_STEP, startPoint);
		}

		// loop
		//#pragma omp parallel for
		for (int j = 1; j < spatialSteps - 1; j = j + 1) {
			data[spatialSteps + j] = data[j] + EQUATION_RATIO * (data[j - 1] - 2 * data[j] + data[j + 1]);
		}

		// j = spatialSteps - 1
		our = data[spatialSteps - 1], left = data[spatialSteps - 2], right;
		if (isInRightRodEdge) {
			right = ENVIROMENT_TEMP;
		} else {
			MPI_Status status;
			MPI_Recv(&right, 1, MPI_DOUBLE, processRank + 1, MPI_SEND_LEFT_EDGE_DATA_BASE_TAG + (i - 1) * 2, MPI_COMM_WORLD, &status);
			//printf("Proc #%d received data (%e) on time %f at point %f\n", processRank + 1, leftData, i * TIME_STEP, endPoint);
		}

		data[2 * spatialSteps - 1] = our + EQUATION_RATIO * (left - 2 * our + right);
		if (!isInRightRodEdge) {
			MPI_Send(data + 2 * spatialSteps - 1, 1, MPI_DOUBLE, processRank + 1, MPI_SEND_RIGHT_EDGE_DATA_BASE_TAG + i * 2, MPI_COMM_WORLD);
			//printf("Proc #%d sent data (%e) on time %f at point %f\n", processRank + 1, *(newData + spatialSteps + j), i * TIME_STEP, startPoint + j * SPATIAL_STEP);
		}

		// После рассылки данных можно сдвинуть массив влево, чтобы предыдущие данные удалить (мы уже всё разослали в блокирующем виде), а текущие превратить в предыдущие и пойти дальше
		double* updatedData = new double[2 * spatialSteps];// static_cast<double*>(calloc(2 * spatialSteps, sizeof(double)));
		for (int k = 0; k < spatialSteps; k = k + 1) {
			updatedData[k] = data[spatialSteps + k];
		}

		delete[] data;//free(data);
		data = updatedData;
	}

	// Высчитываем погрешности для своего участка
	double* preciseData = CalculateRodWithSeries(startPoint, endPoint, 5000);
	double* maxDeviation = new double(std::numeric_limits<double>::min()), *maxDeviationPoint = new double(-1.0);
	for (int i = 0; i < spatialSteps; i = i + 1) {
		double deviation = std::abs(preciseData[i] - data[i]);
		//printf("Deviation for point %f is %e (MPI = %e; serial = %e)\n", startPoint + i * SPATIAL_STEP, deviation, data[i], preciseData[i]);
		if (deviation > *maxDeviation) {
			*maxDeviation = deviation;
			*maxDeviationPoint = startPoint + i * SPATIAL_STEP;
		}
	}

	delete[] preciseData;

	// Перевыделяем память, чтобы не тянуть хвост
	double* resultData = new double[spatialSteps];
	memcpy(resultData, data, spatialSteps * sizeof(double));
	//delete[] data;

	MPI_Barrier(MPI_COMM_WORLD);
	//MPI_Gather(resultData, spatialSteps, MPI_DOUBLE, pointsData, spatialSteps, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Gatherv(resultData, spatialSteps, MPI_DOUBLE, pointsData, gatherCounts, gatherBeginIndices, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	//delete[] resultData;

	MPI_Gather(maxDeviation, 1, MPI_DOUBLE, deviations, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	//delete maxDeviation;

	MPI_Gather(maxDeviationPoint, 1, MPI_DOUBLE, deviationPoints, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	//delete maxDeviationPoint;
}


double* CalculateRodWithSeries(double startPoint, double endPoint, int seriesMembersToCalculate) {
	int spatialSteps = static_cast<int>(round((endPoint - startPoint) / SPATIAL_STEP)) + 1;// static_cast<int>(round(ROD_LENGTH / SPATIAL_STEP)) + 1;
	double* result = new double[spatialSteps];//static_cast<double*>(calloc(spatialSteps, sizeof(double)));
	const double expConstRatio = -CONDUCTIVITY_RATIO * PI * PI * TIME_LENGTH / (ROD_LENGTH * ROD_LENGTH);
	int i;

	//#pragma omp parallel for private(i)
	for (i = 0; i < spatialSteps; i = i + 1) {
		double point = startPoint + i * SPATIAL_STEP;
		double value = 0;

		//#pragma omp parallel for reduction(+:value)
		for (int j = 0; j < seriesMembersToCalculate; j = j + 1) {
			double seriesMemberRatio = 2 * j + 1;;
			value += exp(expConstRatio * seriesMemberRatio * seriesMemberRatio) * sin(PI * seriesMemberRatio * point / ROD_LENGTH) / seriesMemberRatio;
		}

		result[i] = value * 4 * ROD_INITIAL_TEMP / PI;
	}

	return result;
}

