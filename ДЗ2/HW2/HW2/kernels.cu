#include "kernels.cuh"

constexpr int CUDA_STREAMS = 2;
constexpr int BLOCK_SIZE = 2;

void cuda_mm_base(int, int, int, double*, double*, double*&);
void cuda_mm_streams(int, int, int, double*, double*, double*&);
void cuda_mm_sharedmem(int, int, int, double*, double*, double*&);
void cuda_mm_sharedmem2(int, int, int, double*, double*, double*&);
void cuda_mm_sharedmem_streams(int, int, int, double*, double*, double*&);
void cuda_mm_sharedmem2_streams(int, int, int, double*, double*, double*&);
__global__ void mm_kernel_global(int *, int *, int *, double *, double *, double *, int = 0, int = 0);
__global__ void mm_kernel_shared(int *, int *, int *, double *, double *, double *, int = 0, int = 0);
__global__ void mm_kernel_shared2(int *, int *, int *, double *, double *, double *, int = 0, int = 0);


void cuda_mm(int M, int N, int K, double *A, double *B, double *&C)
{
	//cuda_mm_base(M, N, K, A, B, C);
	//cuda_mm_streams(M, N, K, A, B, C);
	//cuda_mm_sharedmem(M, N, K, A, B, C);
	//cuda_mm_sharedmem2(M, N, K, A, B, C);
	cuda_mm_sharedmem_streams(M, N, K, A, B, C);
	//cuda_mm_sharedmem2_streams(M, N, K, A, B, C);
}


void cuda_mm_base(int M, int N, int K, double *A, double *B, double *&C)
{
	printf("Using naive CUDA implementation...\n");
	int *gpuM, *gpuN, *gpuK;
	double *gpuA, *gpuB, *gpuC;

	auto t0 = std::chrono::system_clock::now();
	cudaError_t errorCode;
	errorCode = cudaMalloc(&gpuM, sizeof(int));
	if (errorCode != cudaSuccess) {
		printf("Error occured while allocating M on GPU (%s)", cudaGetErrorString(errorCode));
		return;
	}

	errorCode = cudaMemcpy(gpuM, &M, sizeof(int), cudaMemcpyHostToDevice);
	if (errorCode != cudaSuccess) {
		printf("Error occured while transmitting M to GPU (%s)", cudaGetErrorString(errorCode));
		return;
	}

	errorCode = cudaMalloc(&gpuN, sizeof(int));
	if (errorCode != cudaSuccess) {
		printf("Error occured while allocating N on GPU (%s)", cudaGetErrorString(errorCode));
		return;
	}

	errorCode = cudaMemcpy(gpuN, &N, sizeof(int), cudaMemcpyHostToDevice);
	if (errorCode != cudaSuccess) {
		printf("Error occured while transmitting N to GPU (%s)", cudaGetErrorString(errorCode));
		return;
	}


	errorCode = cudaMalloc(&gpuK, sizeof(int));
	if (errorCode != cudaSuccess) {
		printf("Error occured while allocating K on GPU (%s)", cudaGetErrorString(errorCode));
		return;
	}

	errorCode = cudaMemcpy(gpuK, &K, sizeof(int), cudaMemcpyHostToDevice);
	if (errorCode != cudaSuccess) {
		printf("Error occured while transmitting K to GPU (%s)", cudaGetErrorString(errorCode));
		return;
	}

	errorCode = cudaMalloc(&gpuA, M * N * sizeof(double));
	if (errorCode != cudaSuccess) {
		printf("Error occured while allocating A on GPU (%s)", cudaGetErrorString(errorCode));
		return;
	}

	errorCode = cudaMemcpy(gpuA, A, M * N * sizeof(double), cudaMemcpyHostToDevice);
	if (errorCode != cudaSuccess) {
		printf("Error occured while transmitting A to GPU (%s)", cudaGetErrorString(errorCode));
		return;
	}

	errorCode = cudaMalloc(&gpuB, N * K * sizeof(double));
	if (errorCode != cudaSuccess) {
		printf("Error occured while allocating B on GPU (%s)", cudaGetErrorString(errorCode));
		return;
	}

	errorCode = cudaMemcpy(gpuB, B, N * K * sizeof(double), cudaMemcpyHostToDevice);
	if (errorCode != cudaSuccess) {
		printf("Error occured while transmitting B to GPU (%s)", cudaGetErrorString(errorCode));
		return;
	}

	errorCode = cudaMalloc(&gpuC, M * K * sizeof(double));
	if (errorCode != cudaSuccess) {
		printf("Error occured while allocating C on GPU (%s)", cudaGetErrorString(errorCode));
		return;
	}

	auto t = std::chrono::system_clock::now();
	printf("Intermediate measure. Time complexity for transmitting to GPU: %f s.\n", ConvertChronoToSeconds(t0, t));

	// Оценка геометрии ядра
	dim3 blocks, threads;
	int maxDim = std::max(M, std::max(N, K));
	if (M * K <= 1024) {
		blocks = dim3(1, 1, 1);
		threads = dim3(maxDim, maxDim, 1);
	} else {
		int	mBlocks = M / 32,
			kBlocks = K / 32,
			mResidual = M % 32,
			kResidual = K % 32;
		// Проверку корректности индекса выполняем внутри ядра
		blocks = dim3(mBlocks + std::min(mResidual, 1), kBlocks + std::min(kResidual, 1), 1);
		threads = dim3(32, 32, 1);
	}

	printf("Topology chosen for kernel launch: BLOCKS(%d, %d, 1), THREADS(%d, %d, 1).\n", blocks.x, blocks.y, threads.x, threads.y);

	// Работа ядра

	t0 = std::chrono::system_clock::now();
	mm_kernel_global<<<blocks, threads>>>(gpuM, gpuN, gpuK, gpuA, gpuB, gpuC);
	cudaDeviceSynchronize();
	t = std::chrono::system_clock::now();
	printf("Intermediate measure. Time complexity for kernel execution: %f s.\n", ConvertChronoToSeconds(t0, t));

	errorCode = cudaGetLastError();
	if (errorCode != cudaSuccess) {
		printf("Error occured while executing kernel (%s)", cudaGetErrorString(errorCode));
		return;
	}

	t0 = std::chrono::system_clock::now();
	errorCode = cudaMemcpy(C, gpuC, M * K * sizeof(double), cudaMemcpyDeviceToHost);
	if (errorCode != cudaSuccess) {
		printf("Error occured while transmitting result to RAM (%s)", cudaGetErrorString(errorCode));
		return;
	}

	t = std::chrono::system_clock::now();
	printf("Intermediate measure. Time complexity for transmitting from GPU: %f s.\n", ConvertChronoToSeconds(t0, t));
}


void cuda_mm_streams(int M, int N, int K, double *A, double *B, double *&C)
{
	// Разделяем матрицу на условные зоны ответственности потоков
	int	streamM = floor(M / CUDA_STREAMS),
		streamN = floor(N / CUDA_STREAMS),
		streamK = floor(K / CUDA_STREAMS);
	printf("Using CUDA implementation with async memory allocation...\n");

	int *gpuM, *gpuN, *gpuK;
	double *gpuA, *gpuB, *gpuC;
	cudaError_t errorCode;
	errorCode = cudaMalloc(&gpuM, sizeof(int));
	if (errorCode != cudaSuccess) {
		printf("Error occured while allocating M on GPU (%s)", cudaGetErrorString(errorCode));
		return;
	}

	errorCode = cudaMemcpy(gpuM, &M, sizeof(int), cudaMemcpyHostToDevice);
	if (errorCode != cudaSuccess) {
		printf("Error occured while transmitting M to GPU (%s)", cudaGetErrorString(errorCode));
		return;
	}

	errorCode = cudaMalloc(&gpuN, sizeof(int));
	if (errorCode != cudaSuccess) {
		printf("Error occured while allocating N on GPU (%s)", cudaGetErrorString(errorCode));
		return;
	}
	
	errorCode = cudaMemcpy(gpuN, &N, sizeof(int), cudaMemcpyHostToDevice);
	if (errorCode != cudaSuccess) {
		printf("Error occured while transmitting N to GPU (%s)", cudaGetErrorString(errorCode));
		return;
	}

	errorCode = cudaMalloc(&gpuK, sizeof(int));
	if (errorCode != cudaSuccess) {
		printf("Error occured while allocating K on GPU (%s)", cudaGetErrorString(errorCode));
		return;
	}
	
	errorCode = cudaMemcpy(gpuK, &K, sizeof(int), cudaMemcpyHostToDevice);
	if (errorCode != cudaSuccess) {
		printf("Error occured while transmitting K to GPU (%s)", cudaGetErrorString(errorCode));
		return;
	}

	errorCode = cudaMalloc(&gpuA, M * N * sizeof(double));
	if (errorCode != cudaSuccess) {
		printf("Error occured while allocating A on GPU (%s)", cudaGetErrorString(errorCode));
		return;
	}

	errorCode = cudaMalloc(&gpuB, N * K * sizeof(double));
	if (errorCode != cudaSuccess) {
		printf("Error occured while allocating B on GPU (%s)", cudaGetErrorString(errorCode));
		return;
	}

	errorCode = cudaMalloc(&gpuC, M * K * sizeof(double));
	if (errorCode != cudaSuccess) {
		printf("Error occured while allocating C on GPU (%s)", cudaGetErrorString(errorCode));
		return;
	}

	// Оценка геометрии ядра
	int	mBlocks = M / 32,
		kBlocks = streamK / 32;
	dim3 blocks, threads;
	int maxDim = std::max(M, std::max(N, K));
	if (M * K <= 1024) {
		blocks = dim3(1, 1, 1);
		threads = dim3(maxDim, maxDim, 1);
	} else {
		int	mResidual = M % 32,
			kResidual = streamK % 32;
		// Проверку корректности индекса выполняем внутри ядра
		blocks = dim3(mBlocks + std::min(mResidual, 1), kBlocks + std::min(kResidual, 1), 1);
		threads = dim3(32, 32, 1);
	}
	printf("Topology chosen for kernel launch: STREAMS(%d), BLOCKS(%d, %d, 1), THREADS(%d, %d, 1).\n", CUDA_STREAMS, blocks.x, blocks.y, threads.x, threads.y);

	// Формирование потоков
	printf("EN - Using %d asynchronous CUDA-streams...\n\n", CUDA_STREAMS);
	cudaStream_t streams[CUDA_STREAMS];
	for (int i = 0; i < CUDA_STREAMS; i = i + 1) {
		cudaStreamCreate(&streams[i]);
	}

	// Заполнение потоков операциями переноса в GPU
	int	aStreamStep = (M * N) / CUDA_STREAMS,
		bStreamStep = (N * K) / CUDA_STREAMS,
		cStreamStep = (M * K) / CUDA_STREAMS,
		aStreamResidue = (M * N) % CUDA_STREAMS,
		bStreamResidue = (N * K) % CUDA_STREAMS,
		cStreamResidue = (M * K) % CUDA_STREAMS;

	// Возможно, матрицы не группируются на потоки без остатка
	// В таком случае мы даём последнему потоку дополнительную работу
	// Если остатки нулевые, то получается самый обычный цикл (будто не до [CUDA_STREAMS - 1], а до [CUDA_STREAMS]
	for (int i = 0; i < CUDA_STREAMS - 1; i = i + 1) {
		cudaMemcpyAsync(gpuA + i * aStreamStep,
			A + i * aStreamStep, 
			sizeof(double) * aStreamStep, 
			cudaMemcpyHostToDevice, 
			streams[i]);
		cudaMemcpyAsync(gpuB + i * bStreamStep,
			B + i * bStreamStep,
			sizeof(double) * bStreamStep,
			cudaMemcpyHostToDevice,
			streams[i]);
	}

	cudaMemcpyAsync(gpuA + (CUDA_STREAMS - 1) * aStreamStep,
		A + (CUDA_STREAMS - 1) * aStreamStep,
		sizeof(double) * (aStreamStep + aStreamResidue),
		cudaMemcpyHostToDevice,
		streams[CUDA_STREAMS - 1]);
	cudaMemcpyAsync(gpuB + (CUDA_STREAMS - 1) * bStreamStep,
		B + (CUDA_STREAMS - 1) * bStreamStep,
		sizeof(double) * (bStreamStep + bStreamResidue),
		cudaMemcpyHostToDevice,
		streams[CUDA_STREAMS - 1]);

	// Заполнение потоков вычислениями
	for (int i = 0; i < CUDA_STREAMS; i = i + 1) {
		int columnOffset = i * 32 * blocks.y;
		printf("EN - Stream #%d started with offset %d rows\n", i + 1, columnOffset);
		mm_kernel_global<<<blocks, threads, 0, streams[i]>>>(gpuM, gpuN, gpuK, gpuA, gpuB, gpuC, 0, columnOffset);
	}

	// Заполнение потоков операциями переноса из GPU
	for (int i = 0; i < CUDA_STREAMS - 1; i = i + 1) {
		cudaMemcpyAsync(C + i * cStreamStep,
			gpuC + i * cStreamStep,
			sizeof(double) * cStreamStep,
			cudaMemcpyDeviceToHost,
			streams[i]);
	}

	// См. пояснение выше для матриц А и В
	cudaMemcpyAsync(C + (CUDA_STREAMS - 1) * cStreamStep,
		gpuC + (CUDA_STREAMS - 1) * cStreamStep,
		sizeof(double) * (cStreamStep + cStreamResidue),
		cudaMemcpyDeviceToHost,
		streams[CUDA_STREAMS - 1]);

	cudaDeviceSynchronize();

	// Очищение потоков
	for (int i = 0; i < CUDA_STREAMS; i = i + 1) {
		cudaStreamDestroy(streams[i]);
	}
}


void cuda_mm_sharedmem(int M, int N, int K, double *A, double *B, double *&C)
{
	printf("Using CUDA implementation with tiled multiplication...\n");
	int *gpuM, *gpuN, *gpuK;
	double *gpuA, *gpuB, *gpuC;

	auto t0 = std::chrono::system_clock::now();
	cudaError_t errorCode;
	errorCode = cudaMalloc(&gpuM, sizeof(int));
	if (errorCode != cudaSuccess) {
		printf("Error occured while allocating M on GPU (%s)", cudaGetErrorString(errorCode));
		return;
	}

	errorCode = cudaMemcpy(gpuM, &M, sizeof(int), cudaMemcpyHostToDevice);
	if (errorCode != cudaSuccess) {
		printf("Error occured while transmitting M to GPU (%s)", cudaGetErrorString(errorCode));
		return;
	}

	errorCode = cudaMalloc(&gpuN, sizeof(int));
	if (errorCode != cudaSuccess) {
		printf("Error occured while allocating N on GPU (%s)", cudaGetErrorString(errorCode));
		return;
	}

	errorCode = cudaMemcpy(gpuN, &N, sizeof(int), cudaMemcpyHostToDevice);
	if (errorCode != cudaSuccess) {
		printf("Error occured while transmitting N to GPU (%s)", cudaGetErrorString(errorCode));
		return;
	}


	errorCode = cudaMalloc(&gpuK, sizeof(int));
	if (errorCode != cudaSuccess) {
		printf("Error occured while allocating K on GPU (%s)", cudaGetErrorString(errorCode));
		return;
	}

	errorCode = cudaMemcpy(gpuK, &K, sizeof(int), cudaMemcpyHostToDevice);
	if (errorCode != cudaSuccess) {
		printf("Error occured while transmitting K to GPU (%s)", cudaGetErrorString(errorCode));
		return;
	}

	errorCode = cudaMalloc(&gpuA, M * N * sizeof(double));
	if (errorCode != cudaSuccess) {
		printf("Error occured while allocating A on GPU (%s)", cudaGetErrorString(errorCode));
		return;
	}

	errorCode = cudaMemcpy(gpuA, A, M * N * sizeof(double), cudaMemcpyHostToDevice);
	if (errorCode != cudaSuccess) {
		printf("Error occured while transmitting A to GPU (%s)", cudaGetErrorString(errorCode));
		return;
	}

	errorCode = cudaMalloc(&gpuB, N * K * sizeof(double));
	if (errorCode != cudaSuccess) {
		printf("Error occured while allocating B on GPU (%s)", cudaGetErrorString(errorCode));
		return;
	}

	errorCode = cudaMemcpy(gpuB, B, N * K * sizeof(double), cudaMemcpyHostToDevice);
	if (errorCode != cudaSuccess) {
		printf("Error occured while transmitting B to GPU (%s)", cudaGetErrorString(errorCode));
		return;
	}

	errorCode = cudaMalloc(&gpuC, M * K * sizeof(double));
	if (errorCode != cudaSuccess) {
		printf("Error occured while allocating C on GPU (%s)", cudaGetErrorString(errorCode));
		return;
	}

	auto t = std::chrono::system_clock::now();
	printf("Intermediate measure. Time complexity for transmitting to GPU: %f s.\n", ConvertChronoToSeconds(t0, t));

	// Оценка геометрии ядра
	dim3 blocks, threads;
	int maxDim = std::max(M, std::max(N, K));
	if (maxDim <= 32) {
		blocks = dim3(1, 1, 1);
		threads = dim3(maxDim, maxDim, 1);
	} else {
		int	mBlocks = M / 32,
			kBlocks = K / 32,
			mResidual = M % 32,
			kResidual = K % 32;
		// Проверку корректности индекса выполняем внутри ядра
		blocks = dim3(mBlocks + std::min(mResidual, 1), kBlocks + std::min(kResidual, 1), 1);
		threads = dim3(32, 32, 1);
	}

	printf("Topology chosen for kernel launch: BLOCKS(%d, %d, 1), THREADS(%d, %d, 1).\n", blocks.x, blocks.y, threads.x, threads.y);

	// Работа ядра
	t0 = std::chrono::system_clock::now();
	mm_kernel_shared<<<blocks, threads>>>(gpuM, gpuN, gpuK, gpuA, gpuB, gpuC);
	cudaDeviceSynchronize();
	t = std::chrono::system_clock::now();
	printf("Intermediate measure. Time complexity for kernel execution: %f s.\n", ConvertChronoToSeconds(t0, t));

	errorCode = cudaGetLastError();
	if (errorCode != cudaSuccess) {
		printf("Error occured while executing kernel (%s)", cudaGetErrorString(errorCode));
		return;
	}

	t0 = std::chrono::system_clock::now();
	errorCode = cudaMemcpy(C, gpuC, M * K * sizeof(double), cudaMemcpyDeviceToHost);
	if (errorCode != cudaSuccess) {
		printf("Error occured while transmitting result to RAM (%s)", cudaGetErrorString(errorCode));
		return;
	}

	t = std::chrono::system_clock::now();
	printf("Intermediate measure. Time complexity for transmitting from GPU: %f s.\n", ConvertChronoToSeconds(t0, t));
}


void cuda_mm_sharedmem2(int M, int N, int K, double *A, double *B, double *&C)
{
	printf("Using CUDA implementation with tiled multiplication (4 per thread)...\n");
	int *gpuM, *gpuN, *gpuK;
	double *gpuA, *gpuB, *gpuC;

	auto t0 = std::chrono::system_clock::now();
	cudaError_t errorCode;
	errorCode = cudaMalloc(&gpuM, sizeof(int));
	if (errorCode != cudaSuccess) {
		printf("Error occured while allocating M on GPU (%s)", cudaGetErrorString(errorCode));
		return;
	}

	errorCode = cudaMemcpy(gpuM, &M, sizeof(int), cudaMemcpyHostToDevice);
	if (errorCode != cudaSuccess) {
		printf("Error occured while transmitting M to GPU (%s)", cudaGetErrorString(errorCode));
		return;
	}

	errorCode = cudaMalloc(&gpuN, sizeof(int));
	if (errorCode != cudaSuccess) {
		printf("Error occured while allocating N on GPU (%s)", cudaGetErrorString(errorCode));
		return;
	}

	errorCode = cudaMemcpy(gpuN, &N, sizeof(int), cudaMemcpyHostToDevice);
	if (errorCode != cudaSuccess) {
		printf("Error occured while transmitting N to GPU (%s)", cudaGetErrorString(errorCode));
		return;
	}


	errorCode = cudaMalloc(&gpuK, sizeof(int));
	if (errorCode != cudaSuccess) {
		printf("Error occured while allocating K on GPU (%s)", cudaGetErrorString(errorCode));
		return;
	}

	errorCode = cudaMemcpy(gpuK, &K, sizeof(int), cudaMemcpyHostToDevice);
	if (errorCode != cudaSuccess) {
		printf("Error occured while transmitting K to GPU (%s)", cudaGetErrorString(errorCode));
		return;
	}

	errorCode = cudaMalloc(&gpuA, M * N * sizeof(double));
	if (errorCode != cudaSuccess) {
		printf("Error occured while allocating A on GPU (%s)", cudaGetErrorString(errorCode));
		return;
	}

	errorCode = cudaMemcpy(gpuA, A, M * N * sizeof(double), cudaMemcpyHostToDevice);
	if (errorCode != cudaSuccess) {
		printf("Error occured while transmitting A to GPU (%s)", cudaGetErrorString(errorCode));
		return;
	}

	errorCode = cudaMalloc(&gpuB, N * K * sizeof(double));
	if (errorCode != cudaSuccess) {
		printf("Error occured while allocating B on GPU (%s)", cudaGetErrorString(errorCode));
		return;
	}

	errorCode = cudaMemcpy(gpuB, B, N * K * sizeof(double), cudaMemcpyHostToDevice);
	if (errorCode != cudaSuccess) {
		printf("Error occured while transmitting B to GPU (%s)", cudaGetErrorString(errorCode));
		return;
	}

	errorCode = cudaMalloc(&gpuC, M * K * sizeof(double));
	if (errorCode != cudaSuccess) {
		printf("Error occured while allocating C on GPU (%s)", cudaGetErrorString(errorCode));
		return;
	}

	auto t = std::chrono::system_clock::now();
	printf("Intermediate measure. Time complexity for transmitting to GPU: %f s.\n", ConvertChronoToSeconds(t0, t));

	// Оценка геометрии ядра
	dim3 blocks, threads;
	int maxDim = std::max(M, std::max(N, K));
	if (maxDim <= 32) {
		blocks = dim3(1, 1, 1);
		threads = dim3(maxDim + std::min(maxDim, 1), maxDim, 1);
	} else {
		int	mBlocks = M / 32,
			kBlocks = K / 32,
			mResidual = M % 32,
			kResidual = K % 32;
		// Проверку корректности индекса выполняем внутри ядра
		blocks = dim3(mBlocks + std::min(mResidual, 1), kBlocks + std::min(kResidual, 1), 1);
		threads = dim3(8, 32, 1);
	}

	printf("Topology chosen for kernel launch: BLOCKS(%d, %d, 1), THREADS(%d, %d, 1).\n", blocks.x, blocks.y, threads.x, threads.y);

	// Работа ядра

	t0 = std::chrono::system_clock::now();
	mm_kernel_shared2<<<blocks, threads>>>(gpuM, gpuN, gpuK, gpuA, gpuB, gpuC);
	cudaDeviceSynchronize();
	t = std::chrono::system_clock::now();
	printf("Intermediate measure. Time complexity for kernel execution: %f s.\n", ConvertChronoToSeconds(t0, t));

	errorCode = cudaGetLastError();
	if (errorCode != cudaSuccess) {
		printf("Error occured while executing kernel (%s)", cudaGetErrorString(errorCode));
		return;
	}

	t0 = std::chrono::system_clock::now();
	errorCode = cudaMemcpy(C, gpuC, M * K * sizeof(double), cudaMemcpyDeviceToHost);
	if (errorCode != cudaSuccess) {
		printf("Error occured while transmitting result to RAM (%s)", cudaGetErrorString(errorCode));
		return;
	}

	t = std::chrono::system_clock::now();
	printf("Intermediate measure. Time complexity for transmitting from GPU: %f s.\n", ConvertChronoToSeconds(t0, t));
}


void cuda_mm_sharedmem_streams(int M, int N, int K, double *A, double *B, double *&C)
{
	// Разделяем матрицу на условные зоны ответственности потоков
	int	streamM = floor(M / CUDA_STREAMS) + std::min(M % CUDA_STREAMS, 1),
		streamN = floor(N / CUDA_STREAMS) + std::min(N % CUDA_STREAMS, 1),
		streamK = floor(K / CUDA_STREAMS) + std::min(K % CUDA_STREAMS, 1);

	printf("Using CUDA tiled implementation with async memory allocation...\n");

	int *gpuM, *gpuN, *gpuK;
	double *gpuA, *gpuB, *gpuC;
	cudaError_t errorCode;
	errorCode = cudaMalloc(&gpuM, sizeof(int));
	if (errorCode != cudaSuccess) {
		printf("Error occured while allocating M on GPU (%s)", cudaGetErrorString(errorCode));
		return;
	}

	errorCode = cudaMemcpy(gpuM, &M, sizeof(int), cudaMemcpyHostToDevice);
	if (errorCode != cudaSuccess) {
		printf("Error occured while transmitting M to GPU (%s)", cudaGetErrorString(errorCode));
		return;
	}

	errorCode = cudaMalloc(&gpuN, sizeof(int));
	if (errorCode != cudaSuccess) {
		printf("Error occured while allocating N on GPU (%s)", cudaGetErrorString(errorCode));
		return;
	}

	errorCode = cudaMemcpy(gpuN, &N, sizeof(int), cudaMemcpyHostToDevice);
	if (errorCode != cudaSuccess) {
		printf("Error occured while transmitting N to GPU (%s)", cudaGetErrorString(errorCode));
		return;
	}

	errorCode = cudaMalloc(&gpuK, sizeof(int));
	if (errorCode != cudaSuccess) {
		printf("Error occured while allocating K on GPU (%s)", cudaGetErrorString(errorCode));
		return;
	}

	errorCode = cudaMemcpy(gpuK, &K, sizeof(int), cudaMemcpyHostToDevice);
	if (errorCode != cudaSuccess) {
		printf("Error occured while transmitting K to GPU (%s)", cudaGetErrorString(errorCode));
		return;
	}

	errorCode = cudaMalloc(&gpuA, M * N * sizeof(double));
	if (errorCode != cudaSuccess) {
		printf("Error occured while allocating A on GPU (%s)", cudaGetErrorString(errorCode));
		return;
	}

	errorCode = cudaMalloc(&gpuB, N * K * sizeof(double));
	if (errorCode != cudaSuccess) {
		printf("Error occured while allocating B on GPU (%s)", cudaGetErrorString(errorCode));
		return;
	}

	errorCode = cudaMalloc(&gpuC, M * K * sizeof(double));
	if (errorCode != cudaSuccess) {
		printf("Error occured while allocating C on GPU (%s)", cudaGetErrorString(errorCode));
		return;
	}

	// Оценка геометрии ядра
	int	mBlocks = std::max(M / BLOCK_SIZE, 1),
		kBlocks = std::max(streamK / BLOCK_SIZE, 1);
	dim3 blocks, threads;
	int maxDim = std::max(M, std::max(N, K));
	if (maxDim <= BLOCK_SIZE) {
		blocks = dim3(1, 1, 1);
		threads = dim3(maxDim, maxDim, 1);
	} else {
		int	mResidual = M % BLOCK_SIZE,
			kResidual = streamK % BLOCK_SIZE;
		// Проверку корректности индекса выполняем внутри ядра
		blocks = dim3(mBlocks + std::min(mResidual, 1), kBlocks + std::min(kResidual, 1), 1);
		threads = dim3(BLOCK_SIZE, BLOCK_SIZE, 1);
	}
	printf("Topology chosen for kernel launch: STREAMS(%d), BLOCKS(%d, %d, 1), THREADS(%d, %d, 1).\n", CUDA_STREAMS, blocks.x, blocks.y, threads.x, threads.y);

	// Формирование потоков
	printf("EN - Using %d asynchronous CUDA-streams...\n\n", CUDA_STREAMS);
	cudaStream_t streams[CUDA_STREAMS];
	for (int i = 0; i < CUDA_STREAMS; i = i + 1) {
		cudaStreamCreate(&streams[i]);
	}

	// Заполнение потоков операциями переноса в GPU
	int	aStreamStep = (M * N) / CUDA_STREAMS,
		bStreamStep = (N * K) / CUDA_STREAMS,
		cStreamStep = (M * K) / CUDA_STREAMS,
		aStreamResidue = (M * N) % CUDA_STREAMS,
		bStreamResidue = (N * K) % CUDA_STREAMS,
		cStreamResidue = (M * K) % CUDA_STREAMS;

	// Возможно, матрицы не группируются на потоки без остатка
	// В таком случае мы даём последнему потоку дополнительную работу
	// Если остатки нулевые, то получается самый обычный цикл (будто не до [CUDA_STREAMS - 1], а до [CUDA_STREAMS]
	for (int i = 0; i < CUDA_STREAMS - 1; i = i + 1) {
		cudaMemcpyAsync(gpuA + i * aStreamStep,
			A + i * aStreamStep,
			sizeof(double) * aStreamStep,
			cudaMemcpyHostToDevice,
			streams[i]);
		cudaMemcpyAsync(gpuB + i * bStreamStep,
			B + i * bStreamStep,
			sizeof(double) * bStreamStep,
			cudaMemcpyHostToDevice,
			streams[i]);
	}

	cudaMemcpyAsync(gpuA + (CUDA_STREAMS - 1) * aStreamStep,
		A + (CUDA_STREAMS - 1) * aStreamStep,
		sizeof(double) * (aStreamStep + aStreamResidue),
		cudaMemcpyHostToDevice,
		streams[CUDA_STREAMS - 1]);
	cudaMemcpyAsync(gpuB + (CUDA_STREAMS - 1) * bStreamStep,
		B + (CUDA_STREAMS - 1) * bStreamStep,
		sizeof(double) * (bStreamStep + bStreamResidue),
		cudaMemcpyHostToDevice,
		streams[CUDA_STREAMS - 1]);

	// Заполнение потоков вычислениями
	for (int i = 0; i < CUDA_STREAMS; i = i + 1) {
		int columnOffset = i * BLOCK_SIZE * blocks.y;
		printf("EN - Stream #%d started with offset %d rows\n", i + 1, columnOffset);
		mm_kernel_shared<<<blocks, threads, 0, streams[i]>>>(gpuM, gpuN, gpuK, gpuA, gpuB, gpuC, 0, columnOffset);
	}

	// Заполнение потоков операциями переноса из GPU
	for (int i = 0; i < CUDA_STREAMS - 1; i = i + 1) {
		cudaMemcpyAsync(C + i * cStreamStep,
			gpuC + i * cStreamStep,
			sizeof(double) * cStreamStep,
			cudaMemcpyDeviceToHost,
			streams[i]);
	}

	// См. пояснение выше для матриц А и В
	cudaMemcpyAsync(C + (CUDA_STREAMS - 1) * cStreamStep,
		gpuC + (CUDA_STREAMS - 1) * cStreamStep,
		sizeof(double) * (cStreamStep + cStreamResidue),
		cudaMemcpyDeviceToHost,
		streams[CUDA_STREAMS - 1]);

	cudaDeviceSynchronize();

	errorCode = cudaGetLastError();
	if (errorCode != cudaSuccess) {
		printf("Error occured while streams worked", cudaGetErrorString(errorCode));
		return;
	}

	// Очищение потоков
	for (int i = 0; i < CUDA_STREAMS; i = i + 1) {
		cudaStreamDestroy(streams[i]);
	}
}


void cuda_mm_sharedmem2_streams(int M, int N, int K, double *A, double *B, double *&C)
{
	// Разделяем матрицу на условные зоны ответственности потоков
	int	streamM = floor(M / CUDA_STREAMS),
		streamN = floor(N / CUDA_STREAMS),
		streamK = floor(K / CUDA_STREAMS);
	printf("Using CUDA tiled implementation with async memory allocation (4 per thread)...\n");

	int *gpuM, *gpuN, *gpuK;
	double *gpuA, *gpuB, *gpuC;
	cudaError_t errorCode;
	errorCode = cudaMalloc(&gpuM, sizeof(int));
	if (errorCode != cudaSuccess) {
		printf("Error occured while allocating M on GPU (%s)", cudaGetErrorString(errorCode));
		return;
	}

	errorCode = cudaMemcpy(gpuM, &M, sizeof(int), cudaMemcpyHostToDevice);
	if (errorCode != cudaSuccess) {
		printf("Error occured while transmitting M to GPU (%s)", cudaGetErrorString(errorCode));
		return;
	}

	errorCode = cudaMalloc(&gpuN, sizeof(int));
	if (errorCode != cudaSuccess) {
		printf("Error occured while allocating N on GPU (%s)", cudaGetErrorString(errorCode));
		return;
	}

	errorCode = cudaMemcpy(gpuN, &N, sizeof(int), cudaMemcpyHostToDevice);
	if (errorCode != cudaSuccess) {
		printf("Error occured while transmitting N to GPU (%s)", cudaGetErrorString(errorCode));
		return;
	}

	errorCode = cudaMalloc(&gpuK, sizeof(int));
	if (errorCode != cudaSuccess) {
		printf("Error occured while allocating K on GPU (%s)", cudaGetErrorString(errorCode));
		return;
	}

	errorCode = cudaMemcpy(gpuK, &K, sizeof(int), cudaMemcpyHostToDevice);
	if (errorCode != cudaSuccess) {
		printf("Error occured while transmitting K to GPU (%s)", cudaGetErrorString(errorCode));
		return;
	}

	errorCode = cudaMalloc(&gpuA, M * N * sizeof(double));
	if (errorCode != cudaSuccess) {
		printf("Error occured while allocating A on GPU (%s)", cudaGetErrorString(errorCode));
		return;
	}

	errorCode = cudaMalloc(&gpuB, N * K * sizeof(double));
	if (errorCode != cudaSuccess) {
		printf("Error occured while allocating B on GPU (%s)", cudaGetErrorString(errorCode));
		return;
	}

	errorCode = cudaMalloc(&gpuC, M * K * sizeof(double));
	if (errorCode != cudaSuccess) {
		printf("Error occured while allocating C on GPU (%s)", cudaGetErrorString(errorCode));
		return;
	}

	// Оценка геометрии ядра
	int	mBlocks = M / 32,
		kBlocks = streamK / 32;
	dim3 blocks, threads;
	int maxDim = std::max(M, std::max(N, K));
	if (maxDim <= 32) {
		blocks = dim3(1, 1, 1);
		threads = dim3(maxDim, maxDim, 1);
	} else {
		int	mResidual = M % 32,
			kResidual = streamK % 32;
		// Проверку корректности индекса выполняем внутри ядра
		blocks = dim3(mBlocks + std::min(mResidual, 1), kBlocks + std::min(kResidual, 1), 1);
		threads = dim3(8, 32, 1);
	}
	printf("Topology chosen for kernel launch: STREAMS(%d), BLOCKS(%d, %d, 1), THREADS(%d, %d, 1).\n", CUDA_STREAMS, blocks.x, blocks.y, threads.x, threads.y);

	// Формирование потоков
	printf("EN - Using %d asynchronous CUDA-streams...\n\n", CUDA_STREAMS);
	cudaStream_t streams[CUDA_STREAMS];
	for (int i = 0; i < CUDA_STREAMS; i = i + 1) {
		cudaStreamCreate(&streams[i]);
	}

	// Заполнение потоков операциями переноса в GPU
	int	aStreamStep = (M * N) / CUDA_STREAMS,
		bStreamStep = (N * K) / CUDA_STREAMS,
		cStreamStep = (M * K) / CUDA_STREAMS,
		aStreamResidue = (M * N) % CUDA_STREAMS,
		bStreamResidue = (N * K) % CUDA_STREAMS,
		cStreamResidue = (M * K) % CUDA_STREAMS;

	// Возможно, матрицы не группируются на потоки без остатка
	// В таком случае мы даём последнему потоку дополнительную работу
	// Если остатки нулевые, то получается самый обычный цикл (будто не до [CUDA_STREAMS - 1], а до [CUDA_STREAMS]
	for (int i = 0; i < CUDA_STREAMS - 1; i = i + 1) {
		cudaMemcpyAsync(gpuA + i * aStreamStep,
			A + i * aStreamStep,
			sizeof(double) * aStreamStep,
			cudaMemcpyHostToDevice,
			streams[i]);
		cudaMemcpyAsync(gpuB + i * bStreamStep,
			B + i * bStreamStep,
			sizeof(double) * bStreamStep,
			cudaMemcpyHostToDevice,
			streams[i]);
	}

	cudaMemcpyAsync(gpuA + (CUDA_STREAMS - 1) * aStreamStep,
		A + (CUDA_STREAMS - 1) * aStreamStep,
		sizeof(double) * (aStreamStep + aStreamResidue),
		cudaMemcpyHostToDevice,
		streams[CUDA_STREAMS - 1]);
	cudaMemcpyAsync(gpuB + (CUDA_STREAMS - 1) * bStreamStep,
		B + (CUDA_STREAMS - 1) * bStreamStep,
		sizeof(double) * (bStreamStep + bStreamResidue),
		cudaMemcpyHostToDevice,
		streams[CUDA_STREAMS - 1]);

	// Заполнение потоков вычислениями
	for (int i = 0; i < CUDA_STREAMS; i = i + 1) {
		int columnOffset = i * 32 * blocks.y;
		printf("EN - Stream #%d started with offset %d rows\n", i + 1, columnOffset);
		mm_kernel_shared2<<<blocks, threads, 0, streams[i]>>>(gpuM, gpuN, gpuK, gpuA, gpuB, gpuC, 0, columnOffset);
	}

	// Заполнение потоков операциями переноса из GPU
	for (int i = 0; i < CUDA_STREAMS - 1; i = i + 1) {
		cudaMemcpyAsync(C + i * cStreamStep,
			gpuC + i * cStreamStep,
			sizeof(double) * cStreamStep,
			cudaMemcpyDeviceToHost,
			streams[i]);
	}

	// См. пояснение выше для матриц А и В
	cudaMemcpyAsync(C + (CUDA_STREAMS - 1) * cStreamStep,
		gpuC + (CUDA_STREAMS - 1) * cStreamStep,
		sizeof(double) * (cStreamStep + cStreamResidue),
		cudaMemcpyDeviceToHost,
		streams[CUDA_STREAMS - 1]);

	cudaDeviceSynchronize();

	errorCode = cudaGetLastError();
	if (errorCode != cudaSuccess) {
		printf("Error occured while streams worked (%s)nan", cudaGetErrorString(errorCode));
		return;
	}

	// Очищение потоков
	for (int i = 0; i < CUDA_STREAMS; i = i + 1) {
		cudaStreamDestroy(streams[i]);
	}
}



// CUDA-ядро для перемножения матриц
// Последние два параметра позволяют разбить матрицу на зоны для асинхронных потоков (каждый считает свой кусок матрицы С)
__global__ void mm_kernel_global(int *M, int *N, int *K, double *A, double *B, double *C, int streamRowOffset, int streamColumnOffset)
{
	const int	row = streamRowOffset + blockDim.x * blockIdx.x + threadIdx.x,
			column = streamColumnOffset + blockDim.y * blockIdx.y + threadIdx.y;

	const int m = *M, n = *N, k = *K;
	//if (threadIdx.x % 32 == 31 && threadIdx.y % 32 == 31) printf("(stream #(%f) (thread #(%d, %d)) Working for C[%d, %d]\n", (float) streamColumnOffset / k, blockDim.x * blockIdx.x + threadIdx.x, blockDim.y * blockIdx.y + threadIdx.y, row, column);
	if (row >= m || column >= k) return;

	double	result = C[column * m + row];

	for (int i = 0; i < n; i = i + 1) {
		//if (blockIdx.x > 0 && blockIdx.y > 0 && threadIdx.x == 0 && threadIdx.y == 0) printf("C[%d, %d] (%.3f) += A[%2d, %2d] (%.3f) * B[%2d, %2d] (%.3f)\n", row, column, result, row, i, A[i * m + row], i, column, B[column * n + i]);
		//if (row == 2 && col == 2) printf("(thread #(%d, %d)) i = %d;\nA index : %d, value: %e;\nB index: %d, value: %e.\n\n", blockDim.x * blockIdx.x + threadIdx.x, blockDim.y * blockIdx.y + threadIdx.y, i, i * m + row, A[i * m + row], col * n + i, B[col * n + i]);
		result += A[i * m + row] * B[column * n + i];
	}

	//printf("(thread #(%d, %d)) C index : %d, value: %e.\n\n", blockDim.x * blockIdx.x + threadIdx.x, blockDim.y * blockIdx.y + threadIdx.y, col * m + row, C[col * m + row]);
	C[column * m + row] = result;
}


// CUDA-ядро для перемножения матриц с разделяемой памятью (блочное перемножение)
__global__ void mm_kernel_shared(int *M, int *N, int *K, double *A, double *B, double *C, int streamRowOffset, int streamColumnOffset)
{
	int m = *M, n = *N, k = *K;
	// Блоки определяют кусок матрицы С, который мы обсчитываем
	// Нить - это один из элементов этого блока
	int baseRow = streamRowOffset + BLOCK_SIZE * blockIdx.x, baseColumn = streamColumnOffset + BLOCK_SIZE * blockIdx.y,
		threadRow = baseRow + threadIdx.x, threadColumn = baseColumn + threadIdx.y;
	
	//if (blockIdx.x == 77 && blockIdx.y == 77 && threadIdx.x == 0 && threadIdx.y == 0) printf("(thread #(%d, %d)) Working for C[%d, %d]\n", blockDim.x * blockIdx.x + threadIdx.x, blockDim.y * blockIdx.y + threadIdx.y, threadRow, threadColumn);

	__shared__ double cachedA[BLOCK_SIZE][BLOCK_SIZE + 1];
	__shared__ double cachedB[BLOCK_SIZE][BLOCK_SIZE + 1];

	int div = n - n % BLOCK_SIZE;
	double result = threadRow >= m || threadColumn >= k ? 0.0 : C[threadColumn * m + threadRow];
	// Внешний цикл - это ряд (сумма умножений подматриц)
	for (int i = 0; i < div; i = i + BLOCK_SIZE) {
		// В рамках каждого слагаемого мы остаёмся на том же месте в матрице С, но сдвигаемся вдоль матриц А (вправо по строкам) и B (вниз по столбцам)
		cachedA[threadIdx.x][threadIdx.y] = threadRow >= m ? 0.0 : A[(threadIdx.y + i) * m + threadRow];
		cachedB[threadIdx.x][threadIdx.y] = threadColumn >= k ? 0.0 : B[threadColumn * n + (threadIdx.x + i)];
		__syncthreads();

		// Внутренний цикл - это подсчёт слагаемого ряда (умножение матриц)
		for (int j = 0; j < BLOCK_SIZE; j = j + 1) {
			//if (blockIdx.x == 78 && blockIdx.y == 78 && threadIdx.x == 0 && threadIdx.y == 0) printf("C[%d, %d] (%.3f) += sharedA[%2d, %2d] (%.3f) * sharedb[%2d, %2d] (%.3f)\n", threadRow, threadColumn, result, threadIdx.x, j, cachedA[threadIdx.x][j], j, threadIdx.y, cachedB[j][threadIdx.y]);

			result += cachedA[threadIdx.x][j] * cachedB[j][threadIdx.y];
		}

		__syncthreads();
	}

	if (threadRow >= m || threadColumn >= k) return;

	// То, что не влезает в блоки по 32, считаем по старинке
	for (int i = n % BLOCK_SIZE; i > 0; i = i - 1) {
		//if (blockIdx.x == 78 && blockIdx.y == 78 && threadIdx.x == 0 && threadIdx.y == 0) printf("C[%d, %d] (%.3f) += A[%2d, %2d] (%.3f) * B[%2d, %2d] (%.3f)\n", threadRow, threadColumn, result, threadRow, (div * 32 + i), A[(div * 32 + i) * m + threadRow], (div * 32 + i), threadColumn, B[threadColumn * n + (div * 32 + i)]);
		result += A[(n - i) * m + threadRow] * B[threadColumn * n + (n - i)];
	}

	C[threadColumn * m + threadRow] = result;
}


// CUDA-ядро для перемножения матриц с разделяемой памятью (блочное перемножение куском 32 х 32)
// Здесь один поток отвечает не за один элемент матрицы C, а за несколько
__global__ void mm_kernel_shared2(int *M, int *N, int *K, double *A, double *B, double *C, int streamRowOffset, int streamColumnOffset)
{
	int m = *M, n = *N, k = *K;

	// Блоки по 32х32 определяют кусок матрицы С, который мы обсчитываем
	// Нить - это один из элементов этого блока
	int	threadRow1 = streamRowOffset + 4 * blockDim.x * blockIdx.x + threadIdx.x, threadColumn = streamColumnOffset + blockDim.y * blockIdx.y + threadIdx.y;
	int	threadRow2 = threadRow1 + 8,
		threadRow3 = threadRow2 + 8,
		threadRow4 = threadRow3 + 8;

	//if (blockIdx.x > 0 && blockIdx.y > 0) printf("(thread %d, %d) row2: %d, row3: %d, row4: %d\n", threadRow1, threadColumn, threadRow2, threadRow3, threadRow4);

	__shared__ double cachedA[32][33];
	__shared__ double cachedB[32][33];

	int div = n - n % 32;
	double	result1 = threadRow1 >= m || threadColumn >= k ? 0.0 : C[threadColumn * m + threadRow1],
		result2 = threadRow2 >= m || threadColumn >= k ? 0.0 : C[threadColumn * m + threadRow2],
		result3 = threadRow3 >= m || threadColumn >= k ? 0.0 : C[threadColumn * m + threadRow3],
		result4 = threadRow4 >= m || threadColumn >= k ? 0.0 : C[threadColumn * m + threadRow4];
	// Внешний цикл - это ряд (сумма умножений подматриц)
	for (int i = 0; i < div; i = i + 32) {
		// В рамках каждого слагаемого мы остаёмся на том же месте в матрице С, но сдвигаемся вдоль матриц А (вправо по строкам) и B (вниз по столбцам)

		cachedA[threadIdx.x][threadIdx.y] = threadRow1 >= m ? 0.0 : A[(threadIdx.y + i) * m + threadRow1];
		cachedB[threadIdx.x][threadIdx.y] = threadColumn >= k ? 0.0 : B[threadColumn * n + (threadIdx.x + i)];
		cachedA[threadIdx.x + 8][threadIdx.y] = threadRow2 >= m ? 0.0 : A[(threadIdx.y + i) * m + threadRow2];
		cachedB[threadIdx.x + 8][threadIdx.y] = threadColumn >= k ? 0.0 : B[threadColumn * n + (threadIdx.x + 8 + i)];
		cachedA[threadIdx.x + 16][threadIdx.y] = threadRow3 >= m ? 0.0 : A[(threadIdx.y + i) * m + threadRow3];
		cachedB[threadIdx.x + 16][threadIdx.y] = threadColumn >= k ? 0.0 : B[threadColumn * n + (threadIdx.x + 16 + i)];
		cachedA[threadIdx.x + 24][threadIdx.y] = threadRow4 >= m ? 0.0 : A[(threadIdx.y + i) * m + threadRow4];
		cachedB[threadIdx.x + 24][threadIdx.y] = threadColumn >= k ? 0.0 : B[threadColumn * n + (threadIdx.x + 24 + i)];
		__syncthreads();

		// Внутренний цикл - это подсчёт слагаемого ряда (умножение матриц)
		//if (threadColumn < k) {
		for (int j = 0; j < 32; j = j + 1) {
			//if (/*blockIdx.x > 0 && blockIdx.y > 0 && */threadIdx.x == 0 && threadIdx.y == 0) printf("C[%d, %d] (%.3f) += sharedA[%2d, %2d] (%.3f) * sharedb[%2d, %2d] (%.3f)\n", threadRow, threadColumn, result, threadIdx.x, j, cachedA[threadIdx.x][j], j, threadIdx.y, cachedB[j][threadIdx.y]);

			result1 += cachedA[threadIdx.x][j] * cachedB[j][threadIdx.y];
			result2 += cachedA[threadIdx.x + 8][j] * cachedB[j][threadIdx.y];
			result3 += cachedA[threadIdx.x + 16][j] * cachedB[j][threadIdx.y];
			result4 += cachedA[threadIdx.x + 24][j] * cachedB[j][threadIdx.y];
		}

		__syncthreads();
	}

	if (threadColumn >= k) return;

	// То, что не влезает в блоки по 32, считаем по старинке
	for (int i = n % 32; i > 0; i = i - 1) {
		//if (/*blockIdx.x > 0 && blockIdx.y > 0 && */threadIdx.x == 0 && threadIdx.y == 0) printf("C[%d, %d] (%.3f) += A[%2d, %2d] (%.3f) * B[%2d, %2d] (%.3f)\n", threadRow, threadColumn, result, threadRow, (div * 32 + i), A[(div * 32 + i) * m + threadRow], (div * 32 + i), threadColumn, B[threadColumn * n + (div * 32 + i)]);
		if (threadRow1 < m) result1 += A[(n - i) * m + threadRow1] * B[threadColumn * n + (n - i)];
		if (threadRow2 < m) result2 += A[(n - i) * m + threadRow2] * B[threadColumn * n + (n - i)];
		if (threadRow3 < m) result3 += A[(n - i) * m + threadRow3] * B[threadColumn * n + (n - i)];
		if (threadRow4 < m) result4 += A[(n - i) * m + threadRow4] * B[threadColumn * n + (n - i)];
	}

	if (threadRow1 < m) C[threadColumn * m + threadRow1] = result1;
	if (threadRow2 < m) C[threadColumn * m + threadRow2] = result2;
	if (threadRow3 < m) C[threadColumn * m + threadRow3] = result3;
	if (threadRow4 < m) C[threadColumn * m + threadRow4] = result4;
}