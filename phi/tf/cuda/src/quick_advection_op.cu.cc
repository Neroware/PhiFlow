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


/* =========================================== Density Advection =====================================*/

__global__ void upwindDensityQuickX(float* output_field, float* rho, float* u, int dim, int padding){
    int i = CUDA_THREAD_COL;
    int j = CUDA_THREAD_ROW;

    if(i >= dim + 1 || j >= dim){
        return;
    }
 
    float vel_u = u[pidx(j, i, dim + 1, padding)];
    if(vel_u == 0.0f){
        output_field[IDX(j, i, dim + 1)] = 0.0f;
    }
    else if(vel_u > 0.0f){
        float rho_R, rho_C, rho_L;
        rho_R = rho[pidx(j, i, dim, padding)];
        rho_C = rho[pidx(j, i - 1, dim, padding)];
        rho_L = rho[pidx(j, i - 2, dim, padding)];

        output_field[IDX(j, i, dim + 1)] = 0.5f * (rho_C + rho_R) - 0.125f * (rho_L + rho_R - 2.0f * rho_C);
    }
    else{
        float rho_R, rho_C, rho_FR;
        rho_R = rho[pidx(j, i, dim, padding)];
        rho_FR = rho[pidx(j, i + 1, dim, padding)];
        rho_C = rho[pidx(j, i - 1, dim, padding)];

        output_field[IDX(j, i, dim + 1)] = 0.5f * (rho_C + rho_R) - 0.125f * (rho_FR + rho_C - 2.0f * rho_R);
    }
}


__global__ void upwindDensityQuickY(float* output_field, float* rho, float* v, int dim, int padding){
    int i = CUDA_THREAD_COL;
    int j = CUDA_THREAD_ROW;

    if(i >= dim || j >= dim + 1){
        return;
    }

    float vel_v = v[pidx(j, i, dim, padding)];
    if(vel_v == 0.0f){
        output_field[IDX(j, i, dim)] = 0.0f;
    }
    else if(vel_v > 0.0f){
        float rho_R, rho_C, rho_L;
        rho_R = rho[pidx(j, i, dim, padding)];
        rho_C = rho[pidx(j - 1, i, dim, padding)];
        rho_L = rho[pidx(j - 2, i, dim, padding)];

        output_field[IDX(j, i, dim)] = 0.5f * (rho_C + rho_R) - 0.125f * (rho_L + rho_R - 2.0f * rho_C);
    }
    else{
        float rho_R, rho_C, rho_FR;
        rho_R = rho[pidx(j, i, dim, padding)];
        rho_FR = rho[pidx(j + 1, i, dim, padding)];
        rho_C = rho[pidx(j - 1, i, dim, padding)];

        output_field[IDX(j, i, dim)] = 0.5f * (rho_C + rho_R) - 0.125f * (rho_FR + rho_C - 2.0f * rho_R);
    }
}


__global__ void advectDensityExplicitEuler(float* field, float* rho, float* rho_x, float* rho_y, float* u, float* v, int dim, int padding, float dt){
    int i = CUDA_THREAD_COL;
    int j = CUDA_THREAD_ROW;

    if(i >= dim || j >= dim){
        return;
    }

    // Discretize partial derivates
    float rho_x_1 = rho_x[IDX(j, i, dim + 1)];
    float rho_x_2 = rho_x[IDX(j, i + 1, dim + 1)];
    float u_1 = u[pidx(j, i, dim + 1, padding)];
    float u_2 = u[pidx(j, i + 1, dim + 1, padding)];
    float delta_u_rho_delta_x = u_2 * rho_x_2 - u_1 * rho_x_1;
    
    float rho_y_1 = rho_y[IDX(j, i, dim)];
    float rho_y_2 = rho_y[IDX(j + 1, i, dim)];
    float v_1 = v[pidx(j, i, dim, padding)];
    float v_2 = v[pidx(j + 1, i, dim, padding)];
    float delta_v_rho_delta_y = v_2 * rho_y_2 - v_1 * rho_y_1;

    // Solve Advection Equation
    float delta_rho_delta_t = -delta_u_rho_delta_x - delta_v_rho_delta_y;

    // Perform Explicit Euler
    field[IDX(j, i, dim)] = rho[pidx(j, i, dim, padding)] + delta_rho_delta_t * dt;
}


/* ============================================ Velocity Advection =============================================*/

__global__ void interpolateStaggeredVelocityX(float* output_field, float* u, int dim, int padding){
    int i = CUDA_THREAD_COL;
    int j = CUDA_THREAD_ROW;
    
    if(i >= dim + 1 || j >= dim + 1){
        return;
    }

    float vel_u_1, vel_u_2;
    vel_u_1 = u[pidx(j, i, dim + 1, padding)];
    vel_u_2 = u[pidx(j - 1, i, dim + 1, padding)];

    output_field[IDX(j, i, dim + 1)] = 0.5f * (vel_u_1 + vel_u_2); 
}


__global__ void interpolateStaggeredVelocityY(float* output_field, float* v, int dim, int padding) {
    int i = CUDA_THREAD_COL;
    int j = CUDA_THREAD_ROW;

    if (i >= dim + 1 || j >= dim + 1) {
        return;
    }

    float vel_v_1, vel_v_2;
    vel_v_1 = v[pidx(j, i, dim, padding)];
    vel_v_2 = v[pidx(j, i - 1, dim, padding)];
    
    output_field[IDX(j, i, dim + 1)] = 0.5f * (vel_v_1 + vel_v_2);
}


__global__ void upwindStaggeredVelocityQuickX(float* output_field, float* u, float* v, int dim, int padding) {
    int i = CUDA_THREAD_COL;
    int j = CUDA_THREAD_ROW;

    if (i >= dim + 1 || j >= dim + 1) {
        return;
    }

    float vel_v = v[IDX(j, i, dim + 1)];
    if (vel_v > 0.0f) {
        float u_L, u_C, u_R;
        u_C = u[pidx(j - 1, i, dim + 1, padding)];
        u_L = u[pidx(j - 2, i, dim + 1, padding)];
        u_R = u[pidx(j, i, dim + 1, padding)];
        
        output_field[IDX(j, i, dim + 1)] = 0.5f * (u_C + u_R) - 0.125f * (u_L + u_R - 2.0f * u_C);
    }
    else {
        float u_C, u_R, u_FR;
        u_R = u[pidx(j, i, dim + 1, padding)];
        u_FR = u[pidx(j + 1, i, dim + 1, padding)];
        u_C = u[pidx(j - 1, i, dim + 1, padding)];

        output_field[IDX(j, i, dim + 1)] = 0.5f * (u_C + u_R) - 0.125f * (u_FR + u_C - 2.0f * u_R);
    }
}


__global__ void upwindStaggeredVelocityQuickY(float* output_field, float* u, float* v, int dim, int padding){
    int i = CUDA_THREAD_COL;
    int j = CUDA_THREAD_ROW;

    if(i >= dim + 1 || j >= dim + 1){
        return;
    }

    float vel_u = u[IDX(j, i, dim + 1)];
    if (vel_u > 0.0f) {
        float v_L, v_C, v_R;
        v_C = v[pidx(j, i - 1, dim, padding)];
        v_L = v[pidx(j, i - 2, dim, padding)];
        v_R = v[pidx(j, i, dim, padding)];

        output_field[IDX(j, i, dim + 1)] = 0.5f * (v_C + v_R) - 0.125f * (v_L + v_R - 2.0f * v_C);
    }
    else {
        float v_C, v_R, v_FR;
        v_R = v[pidx(j, i, dim, padding)];
        v_FR = v[pidx(j, i + 1, dim, padding)];
        v_C = v[pidx(j, i - 1, dim, padding)];

        output_field[IDX(j, i, dim + 1)] = 0.5f * (v_C + v_R) - 0.125f * (v_FR + v_C - 2.0f * v_R);
    }
}


__global__ void upwindCenteredVelocityQuickX(float* output_field, float* u, int dim, int padding) {
    int i = CUDA_THREAD_COL;
    int j = CUDA_THREAD_ROW;

    if (i >= dim || j >= dim) {
        return;
    }

    float lerped_u = 0.5f * (u[pidx(j, i, dim + 1, padding)] + u[pidx(j, i + 1, dim + 1, padding)]);
    if (lerped_u > 0.0f) {
        float u_L, u_C, u_R;
        u_C = u[pidx(j, i, dim + 1, padding)];
        u_R = u[pidx(j, i + 1, dim + 1, padding)];
        u_L = u[pidx(j, i - 1, dim + 1, padding)];
        
        output_field[IDX(j, i, dim)] = 0.5f * (u_C + u_R) - 0.125f * (u_L + u_R - 2.0f * u_C);
    }
    else {
        float u_C, u_R, u_FR;
        u_C = u[pidx(j, i, dim + 1, padding)];
        u_R = u[pidx(j, i + 1, dim + 1, padding)];
        u_FR = u[pidx(j, i + 2, dim + 1, padding)];
        
        output_field[IDX(j, i, dim)] = 0.5f * (u_C + u_R) - 0.125f * (u_FR + u_C - 2.0f * u_R);
    }
}


__global__ void upwindCenteredVelocityQuickY(float* output_field, float* v, int dim, int padding) {
    int i = CUDA_THREAD_COL;
    int j = CUDA_THREAD_ROW;

    if (i >= dim || j >= dim) {
        return;
    }

    float lerped_v = 0.5f * (v[pidx(j, i, dim, padding)] + v[pidx(j + 1, i, dim, padding)]);
    if (lerped_v > 0.0f) {
        float v_L, v_C, v_R;
        v_C = v[pidx(j, i, dim, padding)];
        v_R = v[pidx(j + 1, i, dim, padding)];
        v_L = v[pidx(j - 1, i, dim, padding)];

        output_field[IDX(j, i, dim)] = 0.5f * (v_C + v_R) - 0.125f * (v_L + v_R - 2.0f * v_C);
    }
    else {
        float v_C, v_R, v_FR;
        v_C = v[pidx(j, i, dim, padding)];
        v_R = v[pidx(j + 1, i, dim, padding)];
        v_FR = v[pidx(j + 2, i, dim, padding)];

        output_field[IDX(j, i, dim)] = 0.5f * (v_C + v_R) - 0.125f * (v_FR + v_C - 2.0f * v_R);
    }
}


__global__ void advectVelocityXExplicitEuler(float* output_field, float* u_staggered, float* v_staggered, float* u_centered, float* u_field, int dim, int padding, float dt) {
    int i = CUDA_THREAD_COL;
    int j = CUDA_THREAD_ROW;

    if (i >= dim + 1 || j >= dim) {
        return;
    }

    float u_1 = u_staggered[IDX(j + 1, i, dim + 1)];
    float u_2 = u_staggered[IDX(j, i, dim + 1)];
    float v_1 = v_staggered[IDX(j + 1, i, dim + 1)];
    float v_2 = v_staggered[IDX(j, i, dim + 1)];
    float delta_v_u_delta_y = v_1 * u_1 - v_2 * u_2;

    float u_3, u_4;
    if (i == 0) {
        u_4 = u_3 = u_centered[IDX(j, i, dim)];
    }
    else if (i == dim) {
        u_4 = u_3 = u_centered[IDX(j, i - 1, dim)];
    }
    else {
        u_3 = u_centered[IDX(j, i - 1, dim)];
        u_4 = u_centered[IDX(j, i, dim)];
    }
    float delta_u_u_delta_x = u_4 * u_4 - u_3 * u_3;

    float delta_u_delta_t = -delta_u_u_delta_x - delta_v_u_delta_y;
    output_field[IDX(j, i, dim + 1)] = u_field[pidx(j, i, dim + 1, padding)] + delta_u_delta_t * dt;
}


__global__ void advectVelocityYExplicitEuler(float* output_field, float* u_staggered, float* v_staggered, float* v_centered, float* v_field, int dim, int padding, float dt){
    int i = CUDA_THREAD_COL;
    int j = CUDA_THREAD_ROW;

    if(i >= dim || j >= dim + 1){
        return;
    }

    float u_1 = u_staggered[IDX(j, i + 1, dim + 1)];
    float u_2 = u_staggered[IDX(j, i, dim + 1)];
    float v_1 = v_staggered[IDX(j, i + 1, dim + 1)];
    float v_2 = v_staggered[IDX(j, i, dim + 1)];
    float delta_u_v_delta_x = u_1 * v_1 - u_2 * v_2;

    float v_3, v_4;
    if(j == 0){
        v_4 = v_3 = v_centered[IDX(j, i, dim)];
    }
    else if(j == dim){
        v_4 = v_3 = v_centered[IDX(j - 1, i, dim)];
    }
    else{
        v_3 = v_centered[IDX(j - 1, i, dim)];
        v_4 = v_centered[IDX(j, i, dim)];
    }
    float delta_v_v_delta_y = v_4 * v_4 - v_3 * v_3;

    float delta_v_delta_t = -delta_u_v_delta_x - delta_v_v_delta_y;
    output_field[IDX(j, i, dim)] = v_field[pidx(j, i, dim, padding)] + delta_v_delta_t * dt;
}


/* ====================================================== Kernels =============================================*/

void dumpArray(float* d_ptr, int dim_x, int dim_y){
    float* ptr = (float*) malloc(dim_x * dim_y * sizeof(float));
    cudaMemcpy(ptr, d_ptr, dim_x * dim_y * sizeof(float), cudaMemcpyDeviceToHost);

    for(int j = 0; j < dim_y; j++){
        for(int i = 0; i < dim_x; i++){
            printf("%f, ", ptr[IDX(j, i, dim_x)]);
        }
        printf("\n");
    }

    printf("============================================\n");
    delete(ptr);
}


void LaunchQuickDensityKernel(float* output_field, const int dimensions, const int padding, const float timestep, const float* rho, const float* u, const float* v){
    const int DIM = dimensions;
    const int PADDING = padding;
    const int BLOCK_DIM = 16;
    const int BLOCK_ROW_COUNT = ((DIM + 1) / BLOCK_DIM) + 1;
    const dim3 BLOCK(BLOCK_DIM, BLOCK_DIM, 1);
    const dim3 GRID(BLOCK_ROW_COUNT, BLOCK_ROW_COUNT, 1);

    // Setup Device Pointers for u, v and rho
    float *d_rho, *d_u, *d_v;
    cudaMalloc(&d_rho, (DIM + 2 * PADDING) * (DIM + 2 * PADDING) * sizeof(float));
    cudaMalloc(&d_u, (DIM + 2 * PADDING + 1) * (DIM + 2 * PADDING) * sizeof(float));
    cudaMalloc(&d_v, (DIM + 2 * PADDING) * (DIM + 2 * PADDING + 1) * sizeof(float));
    cudaMemcpy(d_rho, rho, (DIM + 2 * PADDING) * (DIM + 2 * PADDING) * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_u, u, (DIM + 2 * PADDING + 1) * (DIM + 2 * PADDING) * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, v, (DIM + 2 * PADDING) * (DIM + 2 * PADDING + 1) * sizeof(float), cudaMemcpyHostToDevice);

    // Calculate Staggered Density Grid using QUICK
    float *d_staggered_density_x, *d_staggered_density_y;
    cudaMalloc(&d_staggered_density_x, (DIM + 1) * DIM * sizeof(float));
    cudaMalloc(&d_staggered_density_y, DIM * (DIM + 1) * sizeof(float));
    upwindDensityQuickX CUDA_CALL(GRID, BLOCK) (d_staggered_density_x, d_rho, d_u, DIM, PADDING);
    upwindDensityQuickY CUDA_CALL(GRID, BLOCK) (d_staggered_density_y, d_rho, d_v, DIM, PADDING);
    
    // Perform Advection Step with Explicit Euler Timestep
    float *d_out;
    cudaMalloc(&d_out, DIM * DIM * sizeof(float));
    advectDensityExplicitEuler CUDA_CALL(GRID, BLOCK) (d_out, d_rho, d_staggered_density_x, d_staggered_density_y, d_u, d_v, DIM, PADDING, timestep);
    cudaMemcpy(output_field, d_out, DIM * DIM * sizeof(float), cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_rho);
    cudaFree(d_u);
    cudaFree(d_v);
    cudaFree(d_staggered_density_x);
    cudaFree(d_staggered_density_y);
    cudaFree(d_out);
}


void LaunchQuickVelocityYKernel(float* output_field, const int dimensions, const int padding, const float timestep, const float* u, const float* v) {
    const int DIM = dimensions;
    const int PADDING = padding;
    const int BLOCK_DIM = 16;
    const int BLOCK_ROW_COUNT = ((DIM + 1) / BLOCK_DIM) + 1;
    const dim3 BLOCK(BLOCK_DIM, BLOCK_DIM, 1);
    const dim3 GRID(BLOCK_ROW_COUNT, BLOCK_ROW_COUNT, 1);

    // Setup Device Pointers for u and v
    float* d_u, * d_v;
    cudaMalloc(&d_u, (DIM + 2 * PADDING + 1) * (DIM + 2 * PADDING) * sizeof(float));
    cudaMalloc(&d_v, (DIM + 2 * PADDING) * (DIM + 2 * PADDING + 1) * sizeof(float));
    cudaMemcpy(d_u, u, (DIM + 2 * PADDING + 1) * (DIM + 2 * PADDING) * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, v, (DIM + 2 * PADDING) * (DIM + 2 * PADDING + 1) * sizeof(float), cudaMemcpyHostToDevice);

    /*printf(">>> u:\n");
    dumpArray(d_u, DIM + 2 * PADDING + 1, DIM + 2 * PADDING);
    printf(">>> v:\n");
    dumpArray(d_v, DIM + 2 * PADDING, DIM + 2 * PADDING + 1);*/

    float* d_staggered_u;
    cudaMalloc(&d_staggered_u, (DIM + 1) * (DIM + 1) * sizeof(float));
    interpolateStaggeredVelocityX CUDA_CALL(GRID, BLOCK) (d_staggered_u, d_u, DIM, PADDING);

    float* d_staggered_v;
    cudaMalloc(&d_staggered_v, (DIM + 1) * (DIM + 1) * sizeof(float));
    upwindStaggeredVelocityQuickY CUDA_CALL(GRID, BLOCK) (d_staggered_v, d_staggered_u, d_v, DIM, PADDING);

    float* d_centered_v;
    cudaMalloc(&d_centered_v, DIM * DIM * sizeof(float));
    upwindCenteredVelocityQuickY CUDA_CALL(GRID, BLOCK) (d_centered_v, d_v, DIM, PADDING);

    /*printf(">>> u staggered:\n");
    dumpArray(d_staggered_u, DIM + 1, DIM + 1);
    printf(">>> v staggered:\n");
    dumpArray(d_staggered_v, DIM + 1, DIM + 1);
    printf(">>> v centered:\n");
    dumpArray(d_centered_v, DIM, DIM);*/

    float* d_out;
    cudaMalloc(&d_out, (DIM + 1) * DIM * sizeof(float));
    advectVelocityYExplicitEuler CUDA_CALL(GRID, BLOCK) (d_out, d_staggered_u, d_staggered_v, d_centered_v, d_v, DIM, PADDING, timestep);
    cudaMemcpy(output_field, d_out, (DIM + 1) * DIM * sizeof(float), cudaMemcpyDeviceToHost);

    /*printf(">>> output:\n");
    dumpArray(d_out, DIM, DIM + 1);*/

    // Cleanup
    cudaFree(d_u);
    cudaFree(d_v);
    cudaFree(d_staggered_u);
    cudaFree(d_staggered_v);
    cudaFree(d_centered_v);
    cudaFree(d_out);
}


void LaunchQuickVelocityXKernel(float* output_field, const int dimensions, const int padding, const float timestep, const float* u, const float* v){
    const int DIM = dimensions;
    const int PADDING = padding;
    const int BLOCK_DIM = 16;
    const int BLOCK_ROW_COUNT = ((DIM + 1) / BLOCK_DIM) + 1;
    const dim3 BLOCK(BLOCK_DIM, BLOCK_DIM, 1);
    const dim3 GRID(BLOCK_ROW_COUNT, BLOCK_ROW_COUNT, 1);

    // Setup Device Pointers for u and v
    float* d_u, * d_v;
    cudaMalloc(&d_u, (DIM + 2 * PADDING + 1) * (DIM + 2 * PADDING) * sizeof(float));
    cudaMalloc(&d_v, (DIM + 2 * PADDING) * (DIM + 2 * PADDING + 1) * sizeof(float));
    cudaMemcpy(d_u, u, (DIM + 2 * PADDING + 1) * (DIM + 2 * PADDING) * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, v, (DIM + 2 * PADDING) * (DIM + 2 * PADDING + 1) * sizeof(float), cudaMemcpyHostToDevice);

    /*printf(">>> u:\n");
    dumpArray(d_u, DIM + 2 * PADDING + 1, DIM + 2 * PADDING);
    printf(">>> v:\n");
    dumpArray(d_v, DIM + 2 * PADDING, DIM + 2 * PADDING + 1);*/

    float* d_staggered_v;
    cudaMalloc(&d_staggered_v, (DIM + 1) * (DIM + 1) * sizeof(float));
    interpolateStaggeredVelocityY CUDA_CALL(GRID, BLOCK) (d_staggered_v, d_v, DIM, PADDING);

    float* d_staggered_u;
    cudaMalloc(&d_staggered_u, (DIM + 1) * (DIM + 1) * sizeof(float));
    upwindStaggeredVelocityQuickX CUDA_CALL(GRID, BLOCK) (d_staggered_u, d_u, d_staggered_v, DIM, PADDING);

    float* d_centered_u;
    cudaMalloc(&d_centered_u, DIM * DIM * sizeof(float));
    upwindCenteredVelocityQuickX CUDA_CALL(GRID, BLOCK) (d_centered_u, d_u, DIM, PADDING);

    /*printf(">>> v staggered:\n");
    dumpArray(d_staggered_v, DIM + 1, DIM + 1);
    printf(">>> u staggered:\n");
    dumpArray(d_staggered_u, DIM + 1, DIM + 1);
    printf(">>> u centered:\n");
    dumpArray(d_centered_u, DIM, DIM);*/

    float* d_out;
    cudaMalloc(&d_out, (DIM + 1) * DIM * sizeof(float));
    advectVelocityXExplicitEuler CUDA_CALL(GRID, BLOCK) (d_out, d_staggered_u, d_staggered_v, d_centered_u, d_u, DIM, PADDING, timestep);
    cudaMemcpy(output_field, d_out, (DIM + 1) * DIM * sizeof(float), cudaMemcpyDeviceToHost);

    /*printf(">>> output:\n");
    dumpArray(d_out, DIM + 1, DIM);*/

    // Cleanup
    cudaFree(d_u);
    cudaFree(d_v);
    cudaFree(d_staggered_u);
    cudaFree(d_staggered_v);
    cudaFree(d_centered_u);
    cudaFree(d_out);
}


