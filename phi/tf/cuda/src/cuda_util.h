/**
 * ================================================================
 * CUDA Utility Header containing some useful macros
 * =================================================================
 **/

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#define CUDA_CALL(blocks, threads) <<< blocks , threads >>>
#define CUDA_CALL_MEM(blocks, threads, shared_memory) <<< blocks , threads , shared_memory >>>

#define CUDA_THREAD_ROW blockIdx.y * blockDim.y + threadIdx.y
#define CUDA_THREAD_COL blockIdx.x * blockDim.x + threadIdx.x

#define CUDA_TIME clock()
