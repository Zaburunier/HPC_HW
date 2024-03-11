#include <thread>
#include <iostream>
#include <ctime>


void CalculateSum(int threadNum, int step, long long length, long long *sum)
{
	long long value = *sum;
	for (long long i = threadNum; i < length; i = i + step) {
		//printf("Thread #%d, index %d\n", threadNum, i);
		value += i;
	}

	*sum = value;
}


void Sum2(long long firstIndex, long long lastIndex, long long *sum)
{
	long long value = *sum;
	for (long long i = firstIndex; i < lastIndex; i = i + 1) {
		value += i;
	}

	*sum = value;
}


int main(int argc, char* argv[])
{
	printf("RU - МСМТ221, Забурунов Леонид Вячеславович, МСМТ221, Высокопроизводительные Вычисления, ДЗ #1.\n");
	printf("EN - MSMT221, Zaburunov Leonid V., High-Performance Computations, Homework #1.\n\n");

	int numThreads = static_cast<int>(std::thread::hardware_concurrency());
	long long seriesLength = 10000;

	if (argc == 1) {
		printf("RU - Аргументы командной строки не заданы. Число потоков задаём по числу логических ядер (%d), длину ряда - 10000.\n", numThreads);
		printf("EN - Args not set. Thread count is equal to logical core count (%d), series length is 100.\n", numThreads);
	} else if (argc == 2) {
		numThreads = std::atoll(argv[1]);
		printf("RU - Получено число потоков - %d. Длина ряда установлена в 10000.\n", numThreads);
		printf("EN - Read thread count - %d. Setting series length for 10000.\n\n", numThreads);
	} else {
		numThreads = std::atoi(argv[1]);
		seriesLength = std::atoll(argv[2]);
		printf("RU - Получено число потоков - %d, получена длина ряда - %lld.\n", numThreads, seriesLength);
		printf("EN - Read thread count - %d, series length - %lld.\n\n", numThreads, seriesLength);
	}

	long long sum = 0;

	auto t0 = std::chrono::system_clock::now();
	if (numThreads == 1) {
		CalculateSum(0, 1, seriesLength, &sum);
	} else {
		std::thread *threads = new std::thread[numThreads];
		long long *sums = (long long *)calloc(sizeof(long long), numThreads);

		for (int i = 0; i < numThreads; i = i + 1) {
			*(threads + i) = std::thread(CalculateSum, i, numThreads, seriesLength, sums + i);
			//*(threads + i) = std::thread(Sum2, i / (numThreads - 1), (i + 1) / (numThreads - 1) - 1, sums + i);
		}

		for (int i = 0; i < numThreads; i = i + 1) {
			threads[i].join();
		}

		for (int i = 0; i < numThreads; i = i + 1) {
			sum += sums[i];
		}
	}

	auto t = std::chrono::system_clock::now();

	auto elapsed = (t - t0).count();
	printf("\nRU/EN - Результат сложения // Result: %lld\n", sum);
	printf("RU/EN - Время исполнения операции // Op time complexity: %lld.\n", elapsed);
	return 0;
}
