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


/* =========================================== Density Advection =====================================*/

__global__ void advectDensityQuick(float* output_field, float* rho, float* u, float* v, int dim_x, int dim_y, float delta_x, float delta_y, int padding, float dt) {
    int i = CUDA_THREAD_COL;
    int j = CUDA_THREAD_ROW;

    if (i >= dim_x || j >= dim_y) {
        return;
    }

    float delta_u_rho_delta_x, delta_v_rho_delta_y;
    delta_u_rho_delta_x = delta_v_rho_delta_y = 0.0f;

    float u1, u2;
    u1 = u[pidx(j, i, dim_x + 1, padding)];
    u2 = u[pidx(j, i + 1, dim_x + 1, padding)];
    float cs_u[5];
    coefficients(cs_u, u1, u2);

    delta_u_rho_delta_x =
        cs_u[0] * rho[pidx(j, i - 2, dim_x, padding)] +
        cs_u[1] * rho[pidx(j, i - 1, dim_x, padding)] +
        cs_u[2] * rho[pidx(j, i, dim_x, padding)] +
        cs_u[3] * rho[pidx(j, i + 1, dim_x, padding)] +
        cs_u[4] * rho[pidx(j, i + 2, dim_x, padding)];
    delta_u_rho_delta_x /= delta_x;
    
    float v1, v2;
    v1 = v[pidx(j, i, dim_x, padding)];
    v2 = v[pidx(j + 1, i, dim_x, padding)];
    float cs_v[5]; 
    coefficients(cs_v, v1, v2);

    delta_v_rho_delta_y =
        cs_v[0] * rho[pidx(j - 2, i, dim_x, padding)] +
        cs_v[1] * rho[pidx(j - 1, i, dim_x, padding)] +
        cs_v[2] * rho[pidx(j, i, dim_x, padding)] +
        cs_v[3] * rho[pidx(j + 1, i, dim_x, padding)] +
        cs_v[4] * rho[pidx(j + 2, i, dim_x, padding)];
    delta_v_rho_delta_y /= delta_y;
    
    float delta_rho_delta_t = -delta_u_rho_delta_x - delta_v_rho_delta_y;
    output_field[IDX(j, i, dim_x)] = rho[pidx(j, i, dim_x, padding)] + delta_rho_delta_t * dt;
}


/* ======================================= Velocity Advection =================================================== */

__global__ void advectVelocityYQuick(float* output_field, float* u, float* v, int dim_x, int dim_y, float delta_x, float delta_y, int padding, float dt) {
    int i = CUDA_THREAD_COL;
    int j = CUDA_THREAD_ROW;

    if (i >= dim_x || j >= dim_y + 1) {
        return;
    }

    float lerped_u1 = 0.5f * (u[pidx(j - 1, i, dim_x + 1, padding)] + u[pidx(j, i, dim_x + 1, padding)]);
    float lerped_u2 = 0.5f * (u[pidx(j, i, dim_x + 1, padding)] + u[pidx(j + 1, i, dim_x + 1, padding)]);
    float v1, v2;
    if(lerped_u1 == 0.0f){
        v1 = 0.0f;
    }
    else if (lerped_u1 > 0.0f) {
        float v_L, v_C, v_R;
        v_C = v[pidx(j, i - 1, dim_x, padding)];
        v_L = v[pidx(j, i - 2, dim_x, padding)];
        v_R = v[pidx(j, i, dim_x, padding)];
        v1 = 0.5f * (v_C + v_R) - 0.125f * (v_L + v_R - 2.0f * v_C);
    }
    else {
        float v_C, v_R, v_FR;
        v_R = v[pidx(j, i, dim_x, padding)];
        v_FR = v[pidx(j, i + 1, dim_x, padding)];
        v_C = v[pidx(j, i - 1, dim_x, padding)];
        v1 = 0.5f * (v_C + v_R) - 0.125f * (v_FR + v_C - 2.0f * v_R);
    }
    if(lerped_u2 == 0.0f){
        v2 = 0.0f;
    }
    else if (lerped_u2 > 0.0f) {
        float v_L, v_C, v_R;
        v_C = v[pidx(j, i, dim_x, padding)];
        v_L = v[pidx(j, i - 1, dim_x, padding)];
        v_R = v[pidx(j, i + 1, dim_x, padding)];
        v2 = 0.5f * (v_C + v_R) - 0.125f * (v_L + v_R - 2.0f * v_C);
    }
    else {
        float v_C, v_R, v_FR;
        v_R = v[pidx(j, i + 1, dim_x, padding)];
        v_FR = v[pidx(j, i + 2, dim_x, padding)];
        v_C = v[pidx(j, i, dim_x, padding)];
        v2 = 0.5f * (v_C + v_R) - 0.125f * (v_FR + v_C - 2.0f * v_R);
    }
    float delta_u_v_delta_x = lerped_u2 * v2 - lerped_u1 * v1;
    delta_u_v_delta_x /= delta_x;

    float lerped_v3 = 0.5f * (v[pidx(j - 1, i, dim_x, padding)] + v[pidx(j, i, dim_x, padding)]);
    float lerped_v4 = 0.5f * (v[pidx(j, i, dim_x, padding)] + v[pidx(j + 1, i, dim_x, padding)]);
    float v3, v4;
    if(lerped_v3 == 0.0f){
        v3 = 0.0f;
    }
    else if (lerped_v3 > 0.0f) {
        float v_L, v_C, v_R;
        v_C = v[pidx(j - 1, i, dim_x, padding)];
        v_R = v[pidx(j, i, dim_x, padding)];
        v_L = v[pidx(j - 2, i, dim_x, padding)];
        v3 = 0.5f * (v_C + v_R) - 0.125f * (v_L + v_R - 2.0f * v_C);
    }
    else {
        float v_C, v_R, v_FR;
        v_C = v[pidx(j - 1, i, dim_x, padding)];
        v_R = v[pidx(j, i, dim_x, padding)];
        v_FR = v[pidx(j + 1, i, dim_x, padding)];
        v3 = 0.5f * (v_C + v_R) - 0.125f * (v_FR + v_C - 2.0f * v_R);
    }
    if (lerped_v4 == 0.0f){
        v4 = 0.0f;
    }
    else if (lerped_v4 > 0.0f) {
        float v_L, v_C, v_R;
        v_C = v[pidx(j, i, dim_x, padding)];
        v_R = v[pidx(j + 1, i, dim_x, padding)];
        v_L = v[pidx(j - 1, i, dim_x, padding)];
        v4 = 0.5f * (v_C + v_R) - 0.125f * (v_L + v_R - 2.0f * v_C);
    }
    else {
        float v_C, v_R, v_FR;
        v_C = v[pidx(j, i, dim_x, padding)];
        v_R = v[pidx(j + 1, i, dim_x, padding)];
        v_FR = v[pidx(j + 2, i, dim_x, padding)];
        v4 = 0.5f * (v_C + v_R) - 0.125f * (v_FR + v_C - 2.0f * v_R);
    }
    float delta_v_v_delta_y = v4 * v4 - v3 * v3;
    delta_v_v_delta_y /= delta_y;

    float delta_v_delta_t = -delta_u_v_delta_x - delta_v_v_delta_y;
    output_field[IDX(j, i, dim_x)] = v[pidx(j, i, dim_x, padding)] + delta_v_delta_t * dt;
}


__global__ void advectVelocityXQuick(float* output_field, float* u, float* v, int dim_x, int dim_y, float delta_x, float delta_y, int padding, float dt) {
    int i = CUDA_THREAD_COL;
    int j = CUDA_THREAD_ROW;

    if (i >= dim_x + 1 || j >= dim_y) {
        return;
    }

    float lerped_v1 = 0.5f * (v[pidx(j, i - 1, dim_x, padding)] + v[pidx(j, i, dim_x, padding)]);
    float lerped_v2 = 0.5f * (v[pidx(j, i, dim_x, padding)] + v[pidx(j, i + 1, dim_x, padding)]);
    float u1, u2;
    if(lerped_v1 == 0.0f){
        u1 = 0.0f;
    }
    else if (lerped_v1 > 0.0f) {
        float u_L, u_C, u_R;
        u_C = u[pidx(j - 1, i, dim_x + 1, padding)];
        u_L = u[pidx(j - 2, i, dim_x + 1, padding)];
        u_R = u[pidx(j, i, dim_x + 1, padding)];
        u1 = 0.5f * (u_C + u_R) - 0.125f * (u_L + u_R - 2.0f * u_C);
    }
    else {
        float u_C, u_R, u_FR;
        u_R = u[pidx(j, i, dim_x + 1, padding)];
        u_FR = u[pidx(j + 1, i, dim_x + 1, padding)];
        u_C = u[pidx(j - 1, i, dim_x + 1, padding)];
        u1 = 0.5f * (u_C + u_R) - 0.125f * (u_FR + u_C - 2.0f * u_R);
    }
    if(lerped_v2 == 0.0f){
        u2 = 0.0f;
    }
    else if (lerped_v2 > 0.0f) {
        float u_L, u_C, u_R;
        u_C = u[pidx(j, i, dim_x + 1, padding)];
        u_L = u[pidx(j - 1, i, dim_x + 1, padding)];
        u_R = u[pidx(j + 1, i, dim_x + 1, padding)];
        u2 = 0.5f * (u_C + u_R) - 0.125f * (u_L + u_R - 2.0f * u_C);
    }
    else {
        float u_C, u_R, u_FR;
        u_R = u[pidx(j + 1, i, dim_x + 1, padding)];
        u_FR = u[pidx(j + 2, i, dim_x + 1, padding)];
        u_C = u[pidx(j, i, dim_x + 1, padding)];
        u2 = 0.5f * (u_C + u_R) - 0.125f * (u_FR + u_C - 2.0f * u_R);
    }
    float delta_v_u_delta_y = lerped_v2 * u2 - lerped_v1 * u1;
    delta_v_u_delta_y /= delta_y;

    float lerped_u3 = 0.5f * (u[pidx(j, i - 1, dim_x + 1, padding)] + u[pidx(j, i, dim_x + 1, padding)]);
    float lerped_u4 = 0.5f * (u[pidx(j, i, dim_x + 1, padding)] + u[pidx(j, i + 1, dim_x + 1, padding)]);
    float u3, u4;
    if(lerped_u3 == 0.0f){
        u3 = 0.0f;
    }
    else if (lerped_u3 > 0.0f) {
        float u_L, u_C, u_R;
        u_C = u[pidx(j, i - 1, dim_x + 1, padding)];
        u_R = u[pidx(j, i, dim_x + 1, padding)];
        u_L = u[pidx(j, i - 2, dim_x + 1, padding)];
        u3 = 0.5f * (u_C + u_R) - 0.125f * (u_L + u_R - 2.0f * u_C);
    }
    else {
        float u_C, u_R, u_FR;
        u_C = u[pidx(j, i - 1, dim_x + 1, padding)];
        u_R = u[pidx(j, i, dim_x + 1, padding)];
        u_FR = u[pidx(j, i + 1, dim_x + 1, padding)];
        u3 = 0.5f * (u_C + u_R) - 0.125f * (u_FR + u_C - 2.0f * u_R);
    }
    if (lerped_u4 == 0.0f){
        u4 = 0.0f;
    }
    else if (lerped_u4 > 0.0f) {
        float u_L, u_C, u_R;
        u_C = u[pidx(j, i, dim_x + 1, padding)];
        u_R = u[pidx(j, i + 1, dim_x + 1, padding)];
        u_L = u[pidx(j, i - 1, dim_x + 1, padding)];
        u4 = 0.5f * (u_C + u_R) - 0.125f * (u_L + u_R - 2.0f * u_C);
    }
    else {
        float u_C, u_R, u_FR;
        u_C = u[pidx(j, i, dim_x + 1, padding)];
        u_R = u[pidx(j, i + 1, dim_x + 1, padding)];
        u_FR = u[pidx(j, i + 2, dim_x + 1, padding)];
        u4 = 0.5f * (u_C + u_R) - 0.125f * (u_FR + u_C - 2.0f * u_R);
    }
    float delta_u_u_delta_x = u4 * u4 - u3 * u3;
    delta_u_u_delta_x /= delta_x;

    float delta_u_delta_t = -delta_u_u_delta_x - delta_v_u_delta_y;
    output_field[IDX(j, i, dim_x + 1)] = u[pidx(j, i, dim_x + 1, padding)] + delta_u_delta_t * dt;
}


/* ====================================================== Kernels =============================================*/

void LaunchQuickDensityKernel(float* output_field, const int dim_x, const int dim_y, const float delta_x, const float delta_y, const int padding, const float timestep, const float* rho, const float* u, const float* v) {
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
    float *d_rho, *d_u, *d_v;
    cudaMalloc(&d_rho, (DIM_X + 2 * PADDING) * (DIM_Y + 2 * PADDING) * sizeof(float));
    cudaMalloc(&d_u, (DIM_X + 2 * PADDING + 1) * (DIM_Y + 2 * PADDING) * sizeof(float));
    cudaMalloc(&d_v, (DIM_X + 2 * PADDING) * (DIM_Y + 2 * PADDING + 1) * sizeof(float));
    cudaMemcpy(d_rho, rho, (DIM_X + 2 * PADDING) * (DIM_Y + 2 * PADDING) * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_u, u, (DIM_X + 2 * PADDING + 1) * (DIM_Y + 2 * PADDING) * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, v, (DIM_X + 2 * PADDING) * (DIM_Y + 2 * PADDING + 1) * sizeof(float), cudaMemcpyHostToDevice);

    // Setup output pointer
    float* d_out;
    cudaMalloc(&d_out, DIM_X * DIM_Y * sizeof(float));

    // Advect the field
    advectDensityQuick CUDA_CALL(GRID, BLOCK) (d_out, d_rho, d_u, d_v, DIM_X, DIM_Y, DELTA_X, DELTA_Y, PADDING, DT);
    cudaMemcpy(output_field, d_out, DIM_X * DIM_Y * sizeof(float), cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_rho);
    cudaFree(d_u);
    cudaFree(d_v);
    cudaFree(d_out);
}


void LaunchQuickVelocityYKernel(float* output_field, const int dim_x, const int dim_y, const float delta_x, const float delta_y, const int padding, const float timestep, const float* u, const float* v) {
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

    // Setup Device Pointers for u and v
    float *d_u, *d_v;
    cudaMalloc(&d_u, (DIM_X + 2 * PADDING + 1) * (DIM_Y + 2 * PADDING) * sizeof(float));
    cudaMalloc(&d_v, (DIM_X + 2 * PADDING) * (DIM_Y + 2 * PADDING + 1) * sizeof(float));
    cudaMemcpy(d_u, u, (DIM_X + 2 * PADDING + 1) * (DIM_Y + 2 * PADDING) * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, v, (DIM_X + 2 * PADDING) * (DIM_Y + 2 * PADDING + 1) * sizeof(float), cudaMemcpyHostToDevice);

    float* d_out;
    cudaMalloc(&d_out, DIM_X * (DIM_Y + 1) * sizeof(float));
    advectVelocityYQuick CUDA_CALL(GRID, BLOCK) (d_out, d_u, d_v, DIM_X, DIM_Y, DELTA_X, DELTA_Y, PADDING, DT);
    cudaMemcpy(output_field, d_out, DIM_X * (DIM_Y + 1) * sizeof(float), cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_u);
    cudaFree(d_v);
    cudaFree(d_out);
}


void LaunchQuickVelocityXKernel(float* output_field, const int dim_x, const int dim_y, const float delta_x, const float delta_y, const int padding, const float timestep, const float* u, const float* v) {
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

    // Setup Device Pointers for u and v
    float *d_u, *d_v;
    cudaMalloc(&d_u, (DIM_X + 2 * PADDING + 1) * (DIM_Y + 2 * PADDING) * sizeof(float));
    cudaMalloc(&d_v, (DIM_X + 2 * PADDING) * (DIM_Y + 2 * PADDING + 1) * sizeof(float));
    cudaMemcpy(d_u, u, (DIM_X + 2 * PADDING + 1) * (DIM_Y + 2 * PADDING) * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, v, (DIM_X + 2 * PADDING) * (DIM_Y + 2 * PADDING + 1) * sizeof(float), cudaMemcpyHostToDevice);

    float* d_out;
    cudaMalloc(&d_out, (DIM_X + 1) * DIM_Y * sizeof(float));
    advectVelocityXQuick CUDA_CALL(GRID, BLOCK) (d_out, d_u, d_v, DIM_X, DIM_Y, DELTA_X, DELTA_Y, PADDING, DT);
    cudaMemcpy(output_field, d_out, (DIM_X + 1) * DIM_Y * sizeof(float), cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_u);
    cudaFree(d_v);
    cudaFree(d_out);
}


