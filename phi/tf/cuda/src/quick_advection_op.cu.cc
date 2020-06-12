#include <cuda.h>
#include <cuda_runtime.h>

#define CUDA_CALL(blocks, threads) <<< blocks , threads >>>
#define CUDA_CALL_MEM(blocks, threads, shared_memory) <<< blocks , threads , shared_memory >>>
#define CUDA_THREAD_ROW blockIdx.y * blockDim.y + threadIdx.y
#define CUDA_THREAD_COL blockIdx.x * blockDim.x + threadIdx.x
#define CUDA_TIME clock()
#define IDX(i, j, dim) i * dim + j

#include <cublas_v2.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <numeric>




__global__ void advectDensityFieldQuick(float* output, float* rho, float* u, float* v, int dim, float dt){
    int i = CUDA_THREAD_ROW;
    int j = CUDA_THREAD_COL;

    if(i >= dim || j >= dim){
        return;
    }

    //output[IDX(j, i, dim)] = (float) j;
    
    // X-Direction
    

    
}


void LaunchQuickDensityKernel(float* output_field, const int dimensions, const float timestep, const float* rho, const float* u, const float* v){
    printf("G'Day!\n");

    const int DIM = dimensions; 
    const int BLOCK_DIM = 16;
    const int BLOCKS_ROW_COUNT = DIM / BLOCK_DIM + 1;
    const dim3 BLOCKS(BLOCK_DIM, BLOCK_DIM, 1);
    const dim3 GRID(BLOCKS_ROW_COUNT, BLOCKS_ROW_COUNT, 1);

    float *d_out, *d_rho, *d_u, *d_v;
    cudaMalloc(&d_out, DIM * DIM * sizeof(float));
    cudaMalloc(&d_rho, DIM * DIM * sizeof(float));
    cudaMalloc(&d_u, (DIM + 1) * (DIM + 1) * sizeof(float));
    cudaMalloc(&d_v, (DIM + 1) * (DIM + 1) * sizeof(float));

    cudaMemcpy(d_rho, rho, DIM * DIM * sizeof(float), cudaMemcpyHostToDevice);   
    cudaMemcpy(d_out, rho, DIM * DIM * sizeof(float), cudaMemcpyHostToDevice); 
    cudaMemcpy(d_u, u, (DIM + 1) * (DIM + 1) * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, v, (DIM + 1) * (DIM + 1) * sizeof(float), cudaMemcpyHostToDevice);

    printf(":D\n");
    advectDensityFieldQuick CUDA_CALL(GRID, BLOCKS) (d_out, d_rho, d_u, d_v, DIM, timestep);
    printf("Done!\n");

    cudaMemcpy(output_field, d_out, DIM * DIM * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_out);
    cudaFree(d_rho);
    cudaFree(d_u);
    cudaFree(d_v);
}
