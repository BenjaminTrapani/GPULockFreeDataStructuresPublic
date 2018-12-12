//
// Created by Benjamin Trapani on 12/22/17.
//

#include <array>
#include <iostream>
#include "LockFreeQueue.hpp"
#include "DoubleListBasedQueue.hpp"
#include "DoubleListBasedStack.hpp"

using namespace LockFreeQueueGPU;

using SumElement_t = unsigned long long int;
constexpr size_t numValuesToSum = 512;

using Queue_t = NUCARLockFreeDS::DoubleListBasedQueue<SumElement_t, numValuesToSum, 0>;

#define CHECK_AND_SYNC                                                                                                           \
{                                                                                                                                \
	cudaError_t error = cudaPeekAtLastError();                                                                                   \
	if (error) {                                                                                                                 \
		std::cerr << "Line " << __LINE__ << ": Ran into error running kernel: " << cudaGetErrorString(error) << std::endl;       \
	}                                                                                                                            \
	cudaDeviceSynchronize();                                                                                                     \
	error = cudaPeekAtLastError();                                                                                               \
	if (error) {                                                                                                                 \
		std::cerr << "Line " << __LINE__ << ": Ran into error synchronizing device: " << cudaGetErrorString(error) << std::endl; \
	}                                                                                                                            \
}

__global__ void initQueue(Queue_t** ppSharedQueue) {
	printf("Will allocate queue\n");
	*ppSharedQueue = new Queue_t();
	printf("Allocated queue, now will sync\n");
	__threadfence();
	printf("Synced allocated queue to memory\n");
}

__global__ void addInitialElements(SumElement_t *valuesToSum, Queue_t** ppSharedQueue) {
	Queue_t* sharedQueue = *ppSharedQueue;
	const size_t indexHere = blockIdx.x * blockDim.x + threadIdx.x;
	const SumElement_t valueHere = valuesToSum[indexHere];
	sharedQueue->enqueue(valueHere);
}

__global__ void performReducePass(Queue_t** ppSharedQueue, int* listSize, unsigned int* totalAdditions) {
	SumElement_t val1;
	SumElement_t val2;

	const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	Queue_t* sharedQueue = *ppSharedQueue;
	if (tid < listSize[0] - 1 || (tid == 0 && listSize[0] == 1)) {
		while (true) {
			const bool popped1 = sharedQueue->dequeue(val1);
			const bool popped2 = sharedQueue->dequeue(val2);

			if (popped1 && popped2) {
				const SumElement_t tempResult = val1 + val2;
				atomicAdd(totalAdditions, 1);
				sharedQueue->enqueue(tempResult);
				atomicSub(listSize, 1);
			}
			else {
				if (popped1) {
					sharedQueue->enqueue(val1);
				}
				if (popped2) {
					sharedQueue->enqueue(val2);
				}
				break;
			}
		}
	}
}

__global__ void finalizeResult(Queue_t** ppSharedQueue, SumElement_t* result, unsigned int* totalAdditions) {
	Queue_t* sharedQueue = *ppSharedQueue;
	sharedQueue->dequeue(result[0]);
	result[1] = totalAdditions[0];
	delete *ppSharedQueue;
}

template<size_t blockCount, size_t threadsPerBlock>
void runReduceSum(SumElement_t* valuesToSum,
	SumElement_t *result,
	Queue_t** ppSharedQueue) {

	int* listSize;
	unsigned int* totalAdditions;
	cudaMalloc(&listSize, sizeof(int));
	cudaMalloc(&totalAdditions, sizeof(unsigned int));
	const int initialListSize = blockCount * threadsPerBlock;
	cudaMemset(totalAdditions, 0, sizeof(unsigned int));
	cudaMemcpy(listSize, &initialListSize, sizeof(int), cudaMemcpyHostToDevice);

	initQueue <<<1, 1>>>(ppSharedQueue);
	CHECK_AND_SYNC

	addInitialElements <<<blockCount, threadsPerBlock>>>(valuesToSum, ppSharedQueue);
	CHECK_AND_SYNC

	while (true) {
		performReducePass <<< blockCount, threadsPerBlock >>> (ppSharedQueue, listSize, totalAdditions);
		CHECK_AND_SYNC
		int curListSize = 0;
		cudaMemcpy(&curListSize, listSize, sizeof(int), cudaMemcpyDeviceToHost);
		if (curListSize <= 1) {
			break;
		}
	}
	finalizeResult <<< 1, 1>>>(ppSharedQueue, result, totalAdditions);
	CHECK_AND_SYNC

	cudaFree(listSize);
	cudaFree(totalAdditions);
}

template<size_t arraySize>
std::array<SumElement_t, arraySize> generateValuesToAdd(){
    std::array<SumElement_t, arraySize> result;
    for(size_t i = 0; i < arraySize; i++){
        result[i] = i + 1;
    }
    return result;
}

template<size_t numElements>
void runReduceSumTest(){
	std::cout << "Sizeof Queue_t: " << sizeof(Queue_t) << std::endl;

    const std::array<SumElement_t, numElements> generatedValues = generateValuesToAdd<numElements>();
    constexpr size_t expectedSum = (numElements * (numElements + 1)) / 2;

	if (cudaDeviceSetLimit(cudaLimit::cudaLimitMallocHeapSize, (size_t)pow(2, 32)) != 0) { //~4 GB
		std::cerr << "Failed to resize GPU heap" << std::endl;
	}
	size_t gpuHeapSize;
	if (cudaDeviceGetLimit(&gpuHeapSize, cudaLimit::cudaLimitMallocHeapSize) != 0) {
		std::cerr << "Failed to fetch GPU heap size" << std::endl;
	}
	std::cout << "GPU heap size: " << gpuHeapSize << std::endl;

	SumElement_t* valuesToSumOnDevice;
    if (cudaMalloc((void**) &valuesToSumOnDevice, sizeof(SumElement_t) * numElements) != 0){
        std::cerr << "Failed to allocate memory for values to sum" << std::endl;
    };
    if (cudaMemcpy(valuesToSumOnDevice, &generatedValues[0], sizeof(SumElement_t) * numElements, cudaMemcpyHostToDevice) != 0){
        std::cerr << "Failed to copy generated values to sum to device" << std::endl;
    }

    SumElement_t* sumResult;
    if (cudaMalloc((void**) &sumResult, sizeof(SumElement_t) * 2) != 0) {
        std::cerr << "Failed to allocate device memory for sum result" << std::endl;
    }
    if (cudaMemset(sumResult, 0, sizeof(SumElement_t) * 2) != 0){
        std::cerr << "Failed to zero device result memory" << std::endl;
    }

    Queue_t **ppQueue;
    if (cudaMalloc((void**) &ppQueue, sizeof(Queue_t*)) != 0) {
        std::cerr << "Failed to allocate pointer to queue" << std::endl;
    }

	runReduceSum<16, 32>(valuesToSumOnDevice, sumResult, ppQueue);

    SumElement_t resultOnHost[2];
	if (cudaMemcpy(&resultOnHost, sumResult, sizeof(SumElement_t) * 2, cudaMemcpyDeviceToHost) != 0) {
		std::cerr << "Failed to copy results back to host: " <<cudaGetErrorString(cudaPeekAtLastError()) << std::endl;
	}
    std::cout << "Total additions: " << resultOnHost[1] << std::endl;
    std::cout << "Sum result: " << resultOnHost[0] << std::endl;
    std::cout << "Expected result: " << expectedSum << std::endl;

	cudaFree(ppQueue);
	cudaFree(sumResult);
	cudaFree(valuesToSumOnDevice);
}

int main(int argc, char** argv){
    runReduceSumTest<numValuesToSum>();
    return 0;
}
