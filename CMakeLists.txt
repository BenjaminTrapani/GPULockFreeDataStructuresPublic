cmake_minimum_required(VERSION 3.3)

IF (WIN32)
    set(CMAKE_GENERATOR_TOOLSET "v140") # TODO revert this once CUDA supports v141
ENDIF()

project(GPULockFreeDataStructures)

find_package(CUDA REQUIRED)

include_directories(${CUDA_INCLUDE_DIRS})
link_directories(${CUDA_LIBRARY_DIRS})

set(CUDA_HOST_COMPILATION_CPP ON)
set(CUDA_NVCC_FLAGS -lineinfo)

cuda_add_executable(LockFreeQueueTest Examples/LockFreeQueueTest.cu)
target_include_directories(LockFreeQueueTest PRIVATE GPULockFreeDataStructures/)
set_property(TARGET LockFreeQueueTest PROPERTY CXX_STANDARD 11)
