#include <cuda.h>
#include <cuda_runtime.h>

#define CUDA_CALL(blocks, threads) <<< blocks , threads >>>
#define CUDA_CALL_MEM(blocks, threads, shared_memory) <<< blocks , threads , shared_memory >>>
#define CUDA_THREAD_ROW (blockIdx.y * blockDim.y + threadIdx.y)
#define CUDA_THREAD_COL (blockIdx.x * blockDim.x + threadIdx.x)
#define CUDA_THREAD_ID (blockIdx.x * blockDim.x + threadIdx.x)
#define CUDA_TIME clock()
#define IDX(i, j, dim) (i) * (dim) + (j)

#include <cublas_v2.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <numeric>




__device__ int pidx(int i, int j, int dim, int padding) {
    return (i + padding) * (dim + 2 * padding) + (j + padding);
}


/**
 * v coefficients, u coefficients; Ranging from (j-2,i) to (j+2,i) and (i-2,j) to (i+2,j) respectivly
 * for properties centered on grid
 */
__device__ float coefficients(int idx, float vel1, float vel2) {
    float c[5];
    c[0] = c[1] = c[2] = c[3] = c[4] = 0.0f;

    if (vel1 >= 0 and vel2 >= 0) {
        c[0] = 0.125f * vel1;
        c[1] = -0.125f * vel2 - 0.75f * vel1;
        c[2] = 0.75f * vel2 - 0.375f * vel1;
        c[3] = 0.375f * vel2;
    }

    else if (vel1 <= 0 and vel2 <= 0) {
        c[1] = -0.375f * vel1;
        c[2] = 0.375f * vel2 - 0.75f * vel1;
        c[3] = 0.75f * vel2 + 0.125f * vel1;
        c[4] = -0.125f * vel2;
    }

    else if (vel1 < 0 and vel2 > 0) {
        c[1] = -0.125f * vel2 - 0.375f * vel1;
        c[2] = 0.75f * vel2 - 0.75f * vel1;
        c[3] = 0.375f * vel2 + 0.125f * vel1;
    }

    else {
        c[0] = 0.125f * vel1;
        c[1] = -0.75f * vel1;
        c[2] = 0.375f * vel2 - 0.375 * vel1;
        c[3] = 0.75f * vel2;
        c[4] = -0.125f * vel1;
    }

    return c[idx];
}


__device__ float coefficients_derivative(int idx, float vel1, float vel2) {
    float c[5];
    c[0] = c[1] = c[2] = c[3] = c[4] = 0.0f;

    if (vel1 >= 0 and vel2 >= 0) {
        c[0] = 0.125f;
        c[1] = -0.125f - 0.75f;
        c[2] = 0.75f - 0.375f;
        c[3] = 0.375f;
    }

    else if (vel1 <= 0 and vel2 <= 0) {
        c[1] = -0.375f;
        c[2] = 0.375f - 0.75f;
        c[3] = 0.75f + 0.125f;
        c[4] = -0.125f;
    }

    else if(vel1 < 0 and vel2 > 0) {
        c[1] = -0.125f - 0.375f;
        c[2] = 0.75f - 0.75f;
        c[3] = 0.375f + 0.125f;
    }

    else{
        c[0] = 0.125f;
        c[1] = -0.75f;
        c[2] = 0.375f - 0.375;
        c[3] = 0.75f;
        c[4] = -0.125f;
    }

    return c[idx];
}


__global__ void gradientFieldQuick(float* output_field, float* rho, float* u, float* v, float* loss, int dim, int padding, float dt) {
    int i = CUDA_THREAD_COL;
    int j = CUDA_THREAD_ROW;

    if (i >= dim || j >= dim) {
        return;
    }

    float u1, u2;
    u1 = u[pidx(j, i, dim + 1, padding)];
    u2 = u[pidx(j, i + 1, dim + 1, padding)];
    float v1, v2;
    v1 = v[pidx(j, i, dim, padding)];
    v2 = v[pidx(j + 1, i, dim, padding)];

    float cs_u[5];
    for (int k = 0; k < 5; k++) {
        cs_u[k] = coefficients(k, u1, u2);
    }
    float cs_v[5];
    for (int k = 0; k < 5; k++) {
        cs_v[k] = coefficients(k, v1, v2);
    }

    output_field[pidx(j, i, dim, padding)] = dt * -(cs_u[0] + cs_u[1] + cs_u[2] + cs_u[3] + cs_u[4]) - (cs_v[0] + cs_v[1] + cs_v[2] + cs_v[3] + cs_v[4]) * loss[IDX(j, i, dim)];
}


__global__ void gradientVelocityXQuick(float* output_field, float* rho, float* u, float* v, float* loss, int dim, int padding, float dt) {
    int i = CUDA_THREAD_COL;
    int j = CUDA_THREAD_ROW;

    if (i >= dim + 1 || j >= dim) {
        return;
    }

    float u1, u2;
    u1 = u[pidx(j, i, dim + 1, padding)];
    u2 = u[pidx(j, i + 1, dim + 1, padding)];
    float cs_u[5];
    for (int k = 0; k < 5; k++) {
        cs_u[k] = coefficients_derivative(k, u1, u2);
    }
    
    float loss_grad;
    if(i >= dim){
        loss_grad = loss[IDX(j, i - 1, dim)];
    }
    loss_grad = loss[IDX(j, i, dim)];

    output_field[pidx(j, i, dim + 1, padding)] = dt * -(
        cs_u[0] * rho[pidx(j, i - 2, dim, padding)] + 
        cs_u[1] * rho[pidx(j, i - 1, dim, padding)] +
        cs_u[2] * rho[pidx(j, i, dim, padding)] + 
        cs_u[3] * rho[pidx(j, i + 1, dim, padding)] +
        cs_u[4] * rho[pidx(j, i + 2, dim, padding)]
   ) * loss_grad;
}


__global__ void gradientVelocityYQuick(float* output_field, float* rho, float* u, float* v, float* loss, int dim, int padding, float dt) {
    int i = CUDA_THREAD_COL;
    int j = CUDA_THREAD_ROW;

    if (i >= dim || j >= dim + 1) {
        return;
    }

    float v1, v2;
    v1 = v[pidx(j, i, dim, padding)];
    v2 = v[pidx(j + 1, i, dim, padding)];
    float cs_v[5];
    for (int k = 0; k < 5; k++) {
        cs_v[k] = coefficients_derivative(k, v1, v2);
    }

    float loss_grad;
    if(j >= dim){
        loss_grad = loss[IDX(j - 1, i, dim)];
    }
    loss_grad = loss[IDX(j, i, dim)];

    output_field[pidx(j, i, dim, padding)] = dt * -(
        cs_v[0] * rho[pidx(j - 2, i, dim, padding)] +
        cs_v[1] * rho[pidx(j - 1, i, dim, padding)] +
        cs_v[2] * rho[pidx(j, i, dim, padding)] +
        cs_v[3] * rho[pidx(j + 1, i, dim, padding)] +
        cs_v[4] * rho[pidx(j + 2, i, dim, padding)]
    ) * loss_grad;
}


void LaunchQUICKAdvectionScalarGradientKernel(float* output_grds, float* vel_u_grds, float* vel_v_grds, const int dimensions, const int padding, const float timestep, const float* rho, const float* u, const float* v, const float* loss){
    const int DIM = dimensions;
    const int PADDING = padding;
    const float DT = timestep;
    const int BLOCK_DIM = 16;
    const int BLOCK_ROW_COUNT = ((DIM + 1) / BLOCK_DIM) + 1;
    const dim3 BLOCK(BLOCK_DIM, BLOCK_DIM, 1);
    const dim3 GRID(BLOCK_ROW_COUNT, BLOCK_ROW_COUNT, 1);

    // Setup Device Pointers for u, v and rho
    float *d_rho, *d_u, *d_v, *d_loss;
    cudaMalloc(&d_rho, (DIM + 2 * PADDING) * (DIM + 2 * PADDING) * sizeof(float));
    cudaMalloc(&d_u, (DIM + 2 * PADDING + 1) * (DIM + 2 * PADDING) * sizeof(float));
    cudaMalloc(&d_v, (DIM + 2 * PADDING) * (DIM + 2 * PADDING + 1) * sizeof(float));
    cudaMalloc(&d_loss, DIM * DIM * sizeof(float));
    cudaMemcpy(d_rho, rho, (DIM + 2 * PADDING) * (DIM + 2 * PADDING) * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_u, u, (DIM + 2 * PADDING + 1) * (DIM + 2 * PADDING) * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, v, (DIM + 2 * PADDING) * (DIM + 2 * PADDING + 1) * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_loss, loss, DIM * DIM * sizeof(float), cudaMemcpyHostToDevice);

    // Setup output pointer
    float *d_out_field, *d_out_u, *d_out_v;
    cudaMalloc(&d_out_field, (DIM + 2 * PADDING) * (DIM + 2 * PADDING) * sizeof(float));
    cudaMalloc(&d_out_u, (DIM + 2 * PADDING + 1) * (DIM + 2 * PADDING) * sizeof(float));
    cudaMalloc(&d_out_v, (DIM + 2 * PADDING) * (DIM + 2 * PADDING + 1) * sizeof(float));

    gradientFieldQuick CUDA_CALL(GRID, BLOCK) (d_out_field, d_rho, d_u, d_v, d_loss, DIM, PADDING, DT);
    cudaMemcpy(output_grds, d_out_field, (DIM + 2 * PADDING) * (DIM + 2 * PADDING) * sizeof(float), cudaMemcpyDeviceToHost);

    gradientVelocityXQuick CUDA_CALL(GRID, BLOCK) (d_out_u, d_rho, d_u, d_v, d_loss, DIM, PADDING, DT);
    cudaMemcpy(vel_u_grds, d_out_u, (DIM + 2 * PADDING + 1) * (DIM + 2 * PADDING) * sizeof(float), cudaMemcpyDeviceToHost);

    gradientVelocityYQuick CUDA_CALL(GRID, BLOCK) (d_out_v, d_rho, d_u, d_v, d_loss, DIM, PADDING, DT);
    cudaMemcpy(vel_v_grds, d_out_v, (DIM + 2 * PADDING) * (DIM + 2 * PADDING + 1) * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_rho);
    cudaFree(d_u);
    cudaFree(d_v);
    cudaFree(d_loss);
    cudaFree(d_out_field);
    cudaFree(d_out_u);
    cudaFree(d_out_v);
}
