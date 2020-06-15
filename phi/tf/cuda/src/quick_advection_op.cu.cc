#include <cuda.h>
#include <cuda_runtime.h>

#define CUDA_CALL(blocks, threads) <<< blocks , threads >>>
#define CUDA_CALL_MEM(blocks, threads, shared_memory) <<< blocks , threads , shared_memory >>>
#define CUDA_THREAD_ROW blockIdx.y * blockDim.y + threadIdx.y
#define CUDA_THREAD_COL blockIdx.x * blockDim.x + threadIdx.x
#define CUDA_THREAD_ID blockIdx.x * blockDim.x + threadIdx.x
#define CUDA_TIME clock()
#define IDX(i, j, dim) (i) * (dim) + (j) // <--- This macro is super evil and caused me so much pain!

#include <cublas_v2.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <numeric>


__global__ void upwindDensityQuickX(float* output_field, float* rho, float* u, int dim){
    int i = CUDA_THREAD_COL;
    int j = CUDA_THREAD_ROW;

    if(i >= dim + 1 || j >= dim){
        return;
    }
 
    float vel_u = u[IDX(j, i, dim + 1)];
    if(vel_u == 0.0f){
        output_field[IDX(j, i, dim + 1)] = 0.0f;
    }
    else if(vel_u > 0.0f){
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
        output_field[IDX(j, i, dim + 1)] = 0.5f * (rho_C + rho_R) - 0.125f * (rho_L + rho_R - 2.0f * rho_C);
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
        output_field[IDX(j, i, dim + 1)] = 0.5f * (rho_C + rho_R) - 0.125f * (rho_FR + rho_C - 2.0f * rho_R);
    }
}


__global__ void upwindDensityQuickY(float* output_field, float* rho, float* v, int dim){
    int i = CUDA_THREAD_COL;
    int j = CUDA_THREAD_ROW;

    if(i >= dim || j >= dim + 1){
        return;
    }

    float vel_v = v[IDX(j, i, dim)];
    if(vel_v == 0.0f){
        output_field[IDX(j, i, dim)] = 0.0f;
    }
    else if(vel_v > 0.0f){
        float rho_R, rho_C, rho_L;
        rho_R = rho_C = rho_L = 0.0f;
        if(j < dim){
            rho_R = rho[IDX(j, i, dim)];
        }
        if(j > 0){
            rho_C = rho[IDX(j - 1, i, dim)];
        }
        if(j > 1){
            rho_L = rho[IDX(j - 2, i, dim)];
        }
        output_field[IDX(j, i, dim)] = 0.5f * (rho_C + rho_R) - 0.125f * (rho_L + rho_R - 2.0f * rho_C);
    }
    else{
        float rho_R, rho_C, rho_FR;
        rho_R = rho_C = rho_FR = 0.0f;
        if(j < dim){
            rho_R = rho[IDX(j, i, dim)];
        }
        if(j < dim - 1){
            rho_FR = rho[IDX(j + 1, i, dim)];
        }
        if(j > 0){
            rho_C = rho[IDX(j - 1, i, dim)];
        }
        output_field[IDX(j, i, dim)] = 0.5f * (rho_C + rho_R) - 0.125f * (rho_FR + rho_C - 2.0f * rho_R);
    }
}


__global__ void advectDensityExplicitEuler(float* field, float* rho, float* rho_x, float* rho_y, float* u, float* v, int dim, float dt){
    int i = CUDA_THREAD_COL;
    int j = CUDA_THREAD_ROW;

    if(i >= dim || j >= dim){
        return;
    }

    // Discretize partial derivates
    float rho_x_1 = rho_x[IDX(j, i, dim + 1)];
    float rho_x_2 = rho_x[IDX(j, i + 1, dim + 1)];
    float u_1 = u[IDX(j, i, dim + 1)];
    float u_2 = u[IDX(j, i + 1, dim + 1)];
    float delta_u_rho_delta_x = u_2 * rho_x_2 - u_1 * rho_x_1;
    
    float rho_y_1 = rho_y[IDX(j, i, dim)];
    float rho_y_2 = rho_y[IDX(j + 1, i, dim)];
    float v_1 = v[IDX(j, i, dim)];
    float v_2 = v[IDX(j + 1, i, dim)];
    float delta_v_rho_delta_y = v_2 * rho_y_2 - v_1 * rho_y_1;

    // Solve Advection Equation
    float delta_rho_delta_t = -delta_u_rho_delta_x - delta_v_rho_delta_y;

    // Perform Explicit Euler
    field[IDX(j, i, dim)] = rho[IDX(j, i, dim)] + delta_rho_delta_t * dt;
}


void LaunchQuickDensityKernel(float* output_field, const int dimensions, const float timestep, const float* rho, const float* u, const float* v){
    const int DIM = dimensions;
    const int BLOCK_DIM = 16;
    const int BLOCK_ROW_COUNT = ((DIM + 1) / BLOCK_DIM) + 1;
    const dim3 BLOCK(BLOCK_DIM, BLOCK_DIM, 1);
    const dim3 GRID(BLOCK_ROW_COUNT, BLOCK_ROW_COUNT, 1);

    // Setup Device Pointers for u, v and rho
    float *d_rho, *d_u, *d_v;
    cudaMalloc(&d_rho, DIM * DIM * sizeof(float));
    cudaMalloc(&d_u, (DIM + 1) * DIM * sizeof(float));
    cudaMalloc(&d_v, DIM * (DIM + 1) * sizeof(float));
    cudaMemcpy(d_rho, rho, DIM * DIM * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_u, u, (DIM + 1) * DIM * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, v, DIM * (DIM + 1) * sizeof(float), cudaMemcpyHostToDevice);

    // Calculate Staggered Density Grid using QUICK
    float *d_staggered_density_x, *d_staggered_density_y;
    cudaMalloc(&d_staggered_density_x, (DIM + 1) * DIM * sizeof(float));
    cudaMalloc(&d_staggered_density_y, DIM * (DIM + 1) * sizeof(float));
    upwindDensityQuickX CUDA_CALL(GRID, BLOCK) (d_staggered_density_x, d_rho, d_u, DIM);
    upwindDensityQuickY CUDA_CALL(GRID, BLOCK) (d_staggered_density_y, d_rho, d_v, DIM);
    
    // Perform Advection Step with Explicit Euler Timestep
    float *d_out;
    cudaMalloc(&d_out, DIM * DIM * sizeof(float));
    advectDensityExplicitEuler CUDA_CALL(GRID, BLOCK) (d_out, d_rho, d_staggered_density_x, d_staggered_density_y, d_u, d_v, DIM, timestep);
    cudaMemcpy(output_field, d_out, DIM * DIM * sizeof(float), cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_rho);
    cudaFree(d_u);
    cudaFree(d_v);
    cudaFree(d_staggered_density_x);
    cudaFree(d_staggered_density_y);
    cudaFree(d_out);
}
