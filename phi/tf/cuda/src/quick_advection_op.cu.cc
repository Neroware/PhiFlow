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



/**
 * Calculates upwind Density at (i-0.5,j)
 */
__device__ float getUpwindDensityX(float* rho, float* u, int i, int j, int dim){
    float vel_u = u[IDX(j, i, dim)];

    if(vel_u == 0.0f){
        return 0.0f;
    }    

    if(vel_u > 0.0f){
        float rho_R, rho_C, rho_L;
        rho_R = rho_C = rho_L = 0.0f;
        if(i < dim){
            rho_R = rho[IDX(j, i, dim)];
        }
        if(i > 0){
            rho_C = rho[IDX(j, i - 1, dim)];
        }
        if(i > 1){
            rho_L = rho[IDX(j, i - 2, dim)];
        }
        return 0.5f * (rho_C + rho_R) - 0.125f * (rho_L + rho_R - 2.0f * rho_C);
    }
    else{
        float rho_R, rho_C, rho_FR;
        rho_R = rho_C = rho_FR = 0.0f;
        if(i < dim){
            rho_R = rho[IDX(j, i, dim)];
        }
        if(i < dim - 1){
            rho_FR = rho[IDX(j, i + 1, dim)];
        }
        if(i > 0){
            rho_C = rho[IDX(j, i - 1, dim)];
        }
        return 0.5f * (rho_C + rho_R) - 0.125f * (rho_FR + rho_C - 2.0f * rho_R);
    }
}


__global__ void advectDensityFieldQuick(float* output, float* rho, float* u, float* v, int dim, float dt){
    int i = CUDA_THREAD_ROW;
    int j = CUDA_THREAD_COL;

    if(i >= dim || j >= dim){
        return;
    }
    
    // Calculate Density at (i-0.5,j) and (i+0.5,j)
    // Compare paper by Leonard
    float rho_1 = getUpwindDensityX(rho, u, i, j, dim);
    float rho_2 = getUpwindDensityX(rho, u, i + 1, j, dim);

    float u_1 = u[IDX(j, i, dim)];
    float u_2 = u[IDX(j, i + 1, dim)];
    float delta_u_rho_delta_x = u_2 * rho_2 - u_1 * rho_1;

    // Solve Advection Equation
    float delta_rho_delta_t = -delta_u_rho_delta_x;

    // Perform Explicit Euler
    output[IDX(j, i, dim)] = rho[IDX(j, i, dim)] + delta_rho_delta_t * dt;
}


void LaunchQuickDensityKernel(float* output_field, const int dimensions, const float timestep, const float* rho, const float* u, const float* v){
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

    advectDensityFieldQuick CUDA_CALL(GRID, BLOCKS) (d_out, d_rho, d_u, d_v, DIM, timestep);

    cudaMemcpy(output_field, d_out, DIM * DIM * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_out);
    cudaFree(d_rho);
    cudaFree(d_u);
    cudaFree(d_v);
}
