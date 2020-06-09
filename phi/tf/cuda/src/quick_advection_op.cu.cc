#include "cuda_util.h"

#include <cublas_v2.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <numeric>




__global__ void advectDensityFieldQuick(float* output, float* rho, float* u, float* v, int dim, float dt){

}


void LaunchQuickDensityKernel(float* output_field, const int dimensions, const float timestep, const float* rho, const float* u, const float* v){
    /*for(int i = 0; i < dimensions * dimensions; i++){
        if(i < 8)
            output_field[i] = 4.2f;
        else
            output_field[i] = 0.0f;
    }*/

    const int DIM = dimensions;
    const int BLOCK_DIM = 16;
    const int THREAD_COUNT = BLOCK_DIM * BLOCK_DIM;
    const int BLOCK_COUNT = DIM / THREAD_COUNT;

    float *d_out, *d_rho, *d_u, *d_v;
    cudaMalloc(&d_out, DIM * DIM * sizeof(float));
    cudaMalloc(&d_rho, DIM * DIM * sizeof(float));
    cudaMalloc(&d_u, (DIM + 1) * (DIM + 1) * sizeof(float));
    cudaMalloc(&d_v, (DIM + 1) * (DIM + 1) * sizeof(float));
    cudaMemcpy(d_out, output_field, DIM * DIM * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rho, rho, DIM * DIM * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_u, u, (DIM + 1) * (DIM + 1) * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, v, (DIM + 1) * (DIM + 1) * sizeof(float), cudaMemcpyHostToDevice);

    printf("Starting Calculation on kernel . . .\n");
    advectDensityFieldQuick CUDA_CALL(BLOCK_COUNT, THREAD_COUNT) (d_out, d_rho, d_u, d_v, DIM, timestep);
    printf("Done!\n");
}
