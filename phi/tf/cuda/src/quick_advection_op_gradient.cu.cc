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
__device__ void coefficients(float* c, float vel1, float vel2) {
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
        c[4] = -0.125f * vel2;
    }
}


__device__ void coefficients_derivative_1(float* c, float vel1, float vel2) {
    c[0] = c[1] = c[2] = c[3] = c[4] = 0.0f;

    if (vel1 >= 0 and vel2 >= 0) {
        c[0] = 0.125f;
        c[1] = -0.75f;
        c[2] = -0.375f;
    }

    else if (vel1 <= 0 and vel2 <= 0) {
        c[1] = -0.375f;
        c[2] = -0.75f;
        c[3] = 0.125f;
    }

    else if (vel1 < 0 and vel2 > 0) {
        c[1] = -0.375f;
        c[2] = -0.75f;
        c[3] = 0.125f;
    }

    else {
        c[0] = 0.125f;
        c[1] = -0.75f;
        c[2] = -0.375;
    }
}


__device__ void coefficients_derivative_2(float* c, float vel1, float vel2) {
    c[0] = c[1] = c[2] = c[3] = c[4] = 0.0f;

    if (vel1 >= 0 and vel2 >= 0) {
        c[1] = -0.125f;
        c[2] = 0.75f;
        c[3] = 0.375f;
    }

    else if (vel1 <= 0 and vel2 <= 0) {
        c[2] = 0.375f;
        c[3] = 0.75f;
        c[4] = -0.125f;
    }

    else if (vel1 < 0 and vel2 > 0) {
        c[1] = -0.125f;
        c[2] = 0.75f;
        c[3] = 0.375f;
    }

    else{
        c[2] = 0.375f;
        c[3] = 0.75f;
        c[4] = -0.125f;
    }
}


__global__ void gradientFieldQuick(float* output_field, float* rho, float* u, float* v, float* grad, int dim_x, int dim_y, float delta_x, float delta_y, int padding, float dt) {
    int i = CUDA_THREAD_COL;
    int j = CUDA_THREAD_ROW;

    if (i >= dim_x || j >= dim_y) {
        return;
    }

    float u1, u2, u3, u4, u5, u6;
    u1 = u[pidx(j, i - 2, dim_x + 1, padding)];
    u2 = u[pidx(j, i - 1, dim_x + 1, padding)];
    u3 = u[pidx(j, i, dim_x + 1, padding)];
    u4 = u[pidx(j, i + 1, dim_x + 1, padding)];
    u5 = u[pidx(j, i + 2, dim_x + 1, padding)];
    u6 = u[pidx(j, i + 3, dim_x + 1, padding)];
    float cs_u1[5];
    coefficients(cs_u1, u1, u2);
    float cs_u2[5]; 
    coefficients(cs_u2, u2, u3);
    float cs_u3[5];
    coefficients(cs_u3, u3, u4);
    float cs_u4[5];
    coefficients(cs_u4, u4, u5);
    float cs_u5[5];
    coefficients(cs_u5, u5, u6);

    float v1, v2, v3, v4, v5, v6;
    v1 = v[pidx(j - 2, i, dim_x, padding)];
    v2 = v[pidx(j - 1, i, dim_x, padding)];
    v3 = v[pidx(j, i, dim_x, padding)];
    v4 = v[pidx(j + 1, i, dim_x, padding)];
    v5 = v[pidx(j + 2, i, dim_x, padding)];
    v6 = v[pidx(j + 3, i, dim_x, padding)];
    float cs_v1[5];
    coefficients(cs_v1, v1, v2);
    float cs_v2[5];
    coefficients(cs_v2, v2, v3);
    float cs_v3[5];
    coefficients(cs_v3, v3, v4);
    float cs_v4[5];
    coefficients(cs_v4, v4, v5);
    float cs_v5[5];
    coefficients(cs_v5, v5, v6);

    float g[9];
    for (int k = 0; k < 9; k++) {
        g[k] = 0.0f;
    }
    g[2] = grad[IDX(j, i, dim_x)];
    if (i > 1)
        g[0] = grad[IDX(j, i - 2, dim_x)];
    if (i > 0)
        g[1] = grad[IDX(j, i - 1, dim_x)];
    if(i < dim - 1)
        g[3] = grad[IDX(j, i + 1, dim_x)];
    if (i < dim - 2)
        g[4] = grad[IDX(j, i + 2, dim_x)];
    if (j > 1)
        g[5] = grad[IDX(j - 2, i, dim_x)];
    if (j > 0)
        g[6] = grad[IDX(j - 1, i, dim_x)];
    if (j < dim - 1)
        g[7] = grad[IDX(j + 1, i, dim_x)];
    if (j < dim - 2)
        g[8] = grad[IDX(j + 2, i, dim_x)];

    float unit_x = 1.0f / delta_x;
    float unit_y = 1.0f / delta_y;

    output_field[pidx(j, i, dim_x, padding)] = g[2] + g[2] * -(unit_x * cs_u3[2] * dt + unit_y * cs_v3[2] * dt) +
        g[0] * unit_x * -(cs_u1[4] * dt) +
        g[1] * unit_x * -(cs_u2[3] * dt) +
        g[3] * unit_x * -(cs_u4[1] * dt) +
        g[4] * unit_x * -(cs_u5[0] * dt) +
        g[5] * unit_y * -(cs_v1[4] * dt) +
        g[6] * unit_y * -(cs_v2[3] * dt) +
        g[7] * unit_y * -(cs_v4[1] * dt) +
        g[8] * unit_y * -(cs_v5[0] * dt);
}


__global__ void gradientVelocityXQuick(float* output_field, float* rho, float* u, float* v, float* grad, int dim_x, int dim_y, float delta_x, int padding, float dt) {
    int i = CUDA_THREAD_COL;
    int j = CUDA_THREAD_ROW;

    if (i >= dim_x + 1 || j >= dim_y) {
        return;
    }

    float u1, u2, u3;
    u1 = u[pidx(j, i - 1, dim_x + 1, padding)];
    u2 = u[pidx(j, i, dim_x + 1, padding)];
    u3 = u[pidx(j, i + 1, dim_x + 1, padding)];

    float cs_u1[5];
    coefficients_derivative_2(cs_u1, u1, u2);
    float cs_u2[5];
    coefficients_derivative_1(cs_u2, u2, u3);

    float grad_1 = 
        cs_u1[0] * rho[pidx(j, i - 3, dim_x, padding)] +
        cs_u1[1] * rho[pidx(j, i - 2, dim_x, padding)] +
        cs_u1[2] * rho[pidx(j, i - 1, dim_x, padding)] +
        cs_u1[3] * rho[pidx(j, i, dim_x, padding)] +
        cs_u1[4] * rho[pidx(j, i + 1, dim_x, padding)];
    grad_1 /= delta_x;

    float grad_2 = 
        cs_u2[0] * rho[pidx(j, i - 2, dim_x, padding)] +
        cs_u2[1] * rho[pidx(j, i - 1, dim_x, padding)] +
        cs_u2[2] * rho[pidx(j, i, dim_x, padding)] +
        cs_u2[3] * rho[pidx(j, i + 1, dim_x, padding)] +
        cs_u2[4] * rho[pidx(j, i + 2, dim_x, padding)];
    grad_2 /= delta_x;

    if(i == 0){
        output_field[pidx(j, i, dim_x + 1, padding)] = -grad_2 * grad[IDX(j, i, dim_x)] * dt;
    }
    else if(i == dim){
        output_field[pidx(j, i, dim_x + 1, padding)] = -grad_1 * grad[IDX(j, i - 1, dim_x)] * dt;
    }
    else{
        output_field[pidx(j, i, dim_x + 1, padding)] = -grad_1 * grad[IDX(j, i - 1, dim_x)] * dt - grad_2 * grad[IDX(j, i, dim_x)] * dt;
    }
}


__global__ void gradientVelocityYQuick(float* output_field, float* rho, float* u, float* v, float* grad, int dim_x, int dim_y, float delta_y, int padding, float dt) {
    int i = CUDA_THREAD_COL;
    int j = CUDA_THREAD_ROW;

    if (i >= dim_x || j >= dim_y + 1) {
        return;
    }

    float v1, v2, v3;
    v1 = v[pidx(j - 1, i, dim_x, padding)];
    v2 = v[pidx(j, i, dim_x, padding)];
    v3 = v[pidx(j + 1, i, dim_x, padding)];

    float cs_v1[5];
    coefficients_derivative_2(cs_v1, v1, v2);
    float cs_v2[5];
    coefficients_derivative_1(cs_v2, v2, v3);

    float grad_1 =
        cs_v1[0] * rho[pidx(j - 3, i, dim_x, padding)] +
        cs_v1[1] * rho[pidx(j - 2, i, dim_x, padding)] +
        cs_v1[2] * rho[pidx(j - 1, i, dim_x, padding)] +
        cs_v1[3] * rho[pidx(j, i, dim_x, padding)] +
        cs_v1[4] * rho[pidx(j + 1, i, dim_x, padding)];
    grad_1 /= delta_y;

    float grad_2 =
        cs_v2[0] * rho[pidx(j - 2, i, dim_x, padding)] +
        cs_v2[1] * rho[pidx(j - 1, i, dim_x, padding)] +
        cs_v2[2] * rho[pidx(j, i, dim_x, padding)] +
        cs_v2[3] * rho[pidx(j + 1, i, dim_x, padding)] +
        cs_v2[4] * rho[pidx(j + 2, i, dim_x, padding)];
    grad_2 /= delta_y;

    if(j == 0){
        output_field[pidx(j, i, dim_x, padding)] = -grad_2 * grad[IDX(j, i, dim_x)] * dt;
    }
    else if(j == dim){
        output_field[pidx(j, i, dim_x, padding)] = -grad_1 * grad[IDX(j - 1, i, dim_x)] * dt;
    }
    else{
        output_field[pidx(j, i, dim_x, padding)] = -grad_1 * grad[IDX(j - 1, i, dim_x)] * dt - grad_2 * grad[IDX(j, i, dim_x)] * dt;
    }
}




void LaunchQUICKAdvectionScalarGradientKernel(float* output_grds, float* vel_u_grds, float* vel_v_grds, const int dim_x, const int dim_y, const float delta_x, const float delta_y, const int padding, const float timestep, const float* rho, const float* u, const float* v, const float* grad){
    const int DIM_X = dim_x;
    const int DIM_Y = dim_y;
    const int DELTA_X = delta_x;
    const int DELTA_Y = delta_y;
    const int PADDING = padding;
    const float DT = timestep;
    const int BLOCK_DIM = 16;
    const int BLOCK_COUNT_X = ((DIM_X + 1) / BLOCK_DIM) + 1;
    const int BLOCK_COUNT_Y = ((DIM_Y + 1) / BLOCK_DIM) + 1;
    const dim3 BLOCK(BLOCK_DIM, BLOCK_DIM, 1);
    const dim3 GRID(BLOCK_COUNT_X, BLOCK_COUNT_Y, 1);

    // Setup Device Pointers for u, v and rho
    float *d_rho, *d_u, *d_v, *d_grad;
    cudaMalloc(&d_rho, (DIM_X + 2 * PADDING) * (DIM_Y + 2 * PADDING) * sizeof(float));
    cudaMalloc(&d_u, (DIM_X + 2 * PADDING + 1) * (DIM_Y + 2 * PADDING) * sizeof(float));
    cudaMalloc(&d_v, (DIM_X + 2 * PADDING) * (DIM_Y + 2 * PADDING + 1) * sizeof(float));
    cudaMalloc(&d_grad, DIM_X * DIM_Y * sizeof(float));
    cudaMemcpy(d_rho, rho, (DIM_X + 2 * PADDING) * (DIM_Y + 2 * PADDING) * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_u, u, (DIM_X + 2 * PADDING + 1) * (DIM_Y + 2 * PADDING) * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, v, (DIM_X + 2 * PADDING) * (DIM_Y + 2 * PADDING + 1) * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_grad, grad, DIM_X * DIM_Y * sizeof(float), cudaMemcpyHostToDevice);

    // Setup output pointer
    float *d_out_field, *d_out_u, *d_out_v;
    cudaMalloc(&d_out_field, (DIM_X + 2 * PADDING) * (DIM_Y + 2 * PADDING) * sizeof(float));
    cudaMalloc(&d_out_u, (DIM_X + 2 * PADDING + 1) * (DIM_Y + 2 * PADDING) * sizeof(float));
    cudaMalloc(&d_out_v, (DIM_X + 2 * PADDING) * (DIM_Y + 2 * PADDING + 1) * sizeof(float));

    gradientFieldQuick CUDA_CALL(GRID, BLOCK) (d_out_field, d_rho, d_u, d_v, d_grad, DIM_X, DIM_Y, DELTA_X, DELTA_Y, PADDING, DT);
    cudaMemcpy(output_grds, d_out_field, (DIM_X + 2 * PADDING) * (DIM_Y + 2 * PADDING) * sizeof(float), cudaMemcpyDeviceToHost);

    gradientVelocityXQuick CUDA_CALL(GRID, BLOCK) (d_out_u, d_rho, d_u, d_v, d_grad, DIM_X, DIM_Y, DELTA_X, DELTA_Y, PADDING, DT);
    cudaMemcpy(vel_u_grds, d_out_u, (DIM_X + 2 * PADDING + 1) * (DIM_Y + 2 * PADDING) * sizeof(float), cudaMemcpyDeviceToHost);

    gradientVelocityYQuick CUDA_CALL(GRID, BLOCK) (d_out_v, d_rho, d_u, d_v, d_grad, DIM_X, DIM_Y, DELTA_X, DELTA_Y, PADDING, DT);
    cudaMemcpy(vel_v_grds, d_out_v, (DIM_X + 2 * PADDING) * (DIM_Y + 2 * PADDING + 1) * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_rho);
    cudaFree(d_u);
    cudaFree(d_v);
    cudaFree(d_grad);
    cudaFree(d_out_field);
    cudaFree(d_out_u);
    cudaFree(d_out_v);
}
