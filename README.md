# GPULockFreeDataStructuresPublic
A header-only collection of lock free data structures that enable sharing data between CUDA threads.

## How to build?
### On Windows:
Run the below in cmd:
```
cmake . -DCMAKE_GENERATOR_PLATFORM=x64
start GPULockFreeDataStructures.sln*
```
Configure LockFreeQueueTest as the startup project in Visual Studio. Then build and run.

### Other platforms:
For Unix, something like this should work:
```bash
cmake .
make
```
## How to use?
A complete example is located in Examples/LockFreeQueueTest.cu. Queue_t can be modified to the desired type to test the other data structures provided. In general, a kernel like this should be used to initialize the chosen data structure on the GPU heap:

```c++
#include "DoubleListBasedQueue.hpp"
__global__ void initQueue(Queue_t** ppSharedQueue)
{
  using TheTypeInMyContainer = long long int;
  constexpr size_t totalThreadCount = 512
  *ppSharedQueue = new NUCARLockFreeDS::DoubleListBasedQueue<TheTypeInMyContainer, totalThreadCount>();
  __threadfence();
}
```

After initialization, it is safe to use ```*ppSharedQueue``` across all threads. The APIs for each data structure are documented in the headers and have markup for doxygen. The doxygen generated docs can be found in the docs folder. The scan threshold used by the garbage collector can be configured via the optional third template parameter to tune performance.

## Limitations and future work
The current garbage collector uses a lot of space for metadata. The space grows at n^2 or more where n is the number of threads. This places a pretty low limit on the number of threads that can concurrently access these data structures. Replacing the existing garbage collector with one that uses much less space for metadata or modifying the existing one to use sparse structures to store the metadata will be a great improvement. 

The current implementation has not been profiled or optimized at all. Profiling the example and implementing backoff per the TODOs in the code will be good first steps towards optimizing these data structures.

The Delete method for the LockFreeDoublyLinkedList has not been tested at all and likely requires bug / compilation fixes.

All contributions for the above items and any others are welcome and much appreciated!
