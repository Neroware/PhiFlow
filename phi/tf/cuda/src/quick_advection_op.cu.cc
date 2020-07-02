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


/* =========================================== Density Advection =====================================*/

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


/* ============================================ Velocity Advection =============================================*/

__global__ void interpolateStaggeredVelocityX(float* output_field, float* u, int dim){
    int i = CUDA_THREAD_COL;
    int j = CUDA_THREAD_ROW;
    
    if(i >= dim + 1 || j >= dim + 1){
        return;
    }

    float vel_u_1, vel_u_2;
    vel_u_1 = vel_u_2 = 0.0f;
    if(j == 0){
        vel_u_1 = vel_u_2 = u[IDX(j, i, dim + 1)];
    }
    else if(j == dim){
        vel_u_1 = vel_u_2 = u[IDX(j - 1, i, dim + 1)];
    }
    else{
        vel_u_1 = u[IDX(j, i, dim + 1)];
        vel_u_2 = u[IDX(j - 1, i, dim + 1)];
    }
    output_field[IDX(j, i, dim + 1)] = 0.5f * (vel_u_1 + vel_u_2); 
}


__global__ void interpolateStaggeredVelocityY(float* output_field, float* v, int dim) {
    int i = CUDA_THREAD_COL;
    int j = CUDA_THREAD_ROW;

    if (i >= dim + 1 || j >= dim + 1) {
        return;
    }

    float vel_v_1, vel_v_2;
    vel_v_1 = vel_v_2 = 0.0f;
    if (i == 0) {
        vel_v_1 = vel_v_2 = v[IDX(j, i, dim)];
    }
    else if (i == dim) {
        vel_v_1 = vel_v_2 = v[IDX(j, i - 1, dim)];
    }
    else {
        vel_v_1 = v[IDX(j, i, dim)];
        vel_v_2 = v[IDX(j, i - 1, dim)];
    }
    output_field[IDX(j, i, dim + 1)] = 0.5f * (vel_v_1 + vel_v_2);
}


__global__ void upwindStaggeredVelocityQuickX(float* output_field, float* u, float* v, int dim) {
    int i = CUDA_THREAD_COL;
    int j = CUDA_THREAD_ROW;

    if (i >= dim + 1 || j >= dim + 1) {
        return;
    }

    float vel_v = v[IDX(j, i, dim + 1)];
    if (vel_v > 0.0f) {
        float u_L, u_C, u_R;
        u_L = u_C = u_R = 0.0f;
        if (j > 0) {
            u_C = u[IDX(j - 1, i, dim + 1)];
        }
        if (j > 1) {
            u_L = u[IDX(j - 2, i, dim + 1)];
        }
        if (j < dim) {
            u_R = u[IDX(j, i, dim + 1)];
        }
        output_field[IDX(j, i, dim + 1)] = 0.5f * (u_C + u_R) - 0.125f * (u_L + u_R - 2.0f * u_C);
    }
    else {
        float u_C, u_R, u_FR;
        u_C = u_R = u_FR = 0.0f;
        if (j < dim) {
            u_R = u[IDX(j, i, dim + 1)];
        }
        if (j < dim - 1) {
            u_FR = u[IDX(j + 1, i, dim + 1)];
        }
        if (j > 0) {
            u_C = u[IDX(j - 1, i, dim + 1)];
        }
        output_field[IDX(j, i, dim + 1)] = 0.5f * (u_C + u_R) - 0.125f * (u_FR + u_C - 2.0f * u_R);
    }
}


__global__ void upwindStaggeredVelocityQuickY(float* output_field, float* u, float* v, int dim){
    int i = CUDA_THREAD_COL;
    int j = CUDA_THREAD_ROW;

    if(i >= dim + 1 || j >= dim + 1){
        return;
    }

    float vel_u = u[IDX(j, i, dim + 1)];
    if (vel_u > 0.0f) {
        float v_L, v_C, v_R;
        v_L = v_C = v_R = 0.0f;
        if(i > 0){
            v_C = v[IDX(j, i - 1, dim)];
        }
        if(i > 1){
            v_L = v[IDX(j, i - 2, dim)];
        }
        if(i < dim){
            v_R = v[IDX(j, i, dim)];
        }
        output_field[IDX(j, i, dim + 1)] = 0.5f * (v_C + v_R) - 0.125f * (v_L + v_R - 2.0f * v_C);
        //output_field[IDX(j, i, dim + 1)] = 0.125f * v_L + 0.25f * v_C + 0.625f * v_R;
    }
    else {
        float v_C, v_R, v_FR;
        v_C = v_R = v_FR = 0.0f;
        if (i < dim) {
            v_R = v[IDX(j, i, dim)];
        }
        if (i < dim - 1) {
            v_FR = v[IDX(j, i + 1, dim)];
        }
        if(i > 0){
            v_C = v[IDX(j, i - 1, dim)];
        }
        output_field[IDX(j, i, dim + 1)] = 0.5f * (v_C + v_R) - 0.125f * (v_FR + v_C - 2.0f * v_R);
        //output_field[IDX(j, i, dim + 1)] = 0.625f * v_C + 0.25f * v_R + 0.125f * v_FR;
    }
}


__global__ void upwindCenteredVelocityQuickX(float* output_field, float* u, int dim) {
    int i = CUDA_THREAD_COL;
    int j = CUDA_THREAD_ROW;

    if (i >= dim || j >= dim) {
        return;
    }

    float lerped_u = 0.5f * (u[IDX(j, i, dim + 1)] + u[IDX(j, i + 1, dim)]);
    if (lerped_u > 0.0f) {
        float u_L, u_C, u_R;
        u_L = u_C = u_R = u[IDX(j, i, dim + 1)];
        u_C = u[IDX(j, i, dim + 1)];
        u_R = u[IDX(j, i + 1, dim + 1)];
        if (i > 0) {
            u_L = u[IDX(j, i - 1, dim + 1)];
        }
        output_field[IDX(j, i, dim)] = 0.5f * (u_C + u_R) - 0.125f * (u_L + u_R - 2.0f * u_C);
    }
    else {
        float u_C, u_R, u_FR;
        u_C = u_R = u_FR = u[IDX(j, i, dim + 1)];
        u_C = u[IDX(j, i, dim + 1)];
        u_R = u[IDX(j, i + 1, dim + 1)];
        if (i < dim) {
            v_FR = u[IDX(j, i + 2, dim + 1)];
        }
        output_field[IDX(j, i, dim)] = 0.5f * (u_C + u_R) - 0.125f * (u_FR + u_C - 2.0f * u_R);
    }
}


__global__ void upwindCenteredVelocityQuickY(float* output_field, float* v, int dim) {
    int i = CUDA_THREAD_COL;
    int j = CUDA_THREAD_ROW;

    if (i >= dim || j >= dim) {
        return;
    }

    float lerped_v = 0.5f * (v[IDX(j, i, dim)] + v[IDX(j + 1, i, dim)]);
    if (lerped_v > 0.0f) {
        float v_L, v_C, v_R;
        v_L = v_C = v_R = v[IDX(j, i, dim)];
        v_C = v[IDX(j, i, dim)];
        v_R = v[IDX(j + 1, i, dim)];
        if (j > 0) {
            v_L = v[IDX(j - 1, i, dim)];
        }
        output_field[IDX(j, i, dim)] = 0.5f * (v_C + v_R) - 0.125f * (v_L + v_R - 2.0f * v_C);
    }
    else {
        float v_C, v_R, v_FR;
        v_C = v_R = v_FR = v[IDX(j, i, dim)];
        v_C = v[IDX(j, i, dim)];
        v_R = v[IDX(j + 1, i, dim)];
        if (j < dim) {
            v_FR = v[IDX(j + 2, i, dim)];
        }
        output_field[IDX(j, i, dim)] = 0.5f * (v_C + v_R) - 0.125f * (v_FR + v_C - 2.0f * v_R);
    }
}


__global__ void advectVelocityXExplicitEuler(float* output_field, float* u_staggered, float* v_staggered, float* u_centered, float* u_field, int dim, float dt) {
    int i = CUDA_THREAD_COL;
    int j = CUDA_THREAD_ROW;

    if (i >= dim || j >= dim + 1) {
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
    output_field[IDX(j, i, dim + 1)] = u_field[IDX(j, i, dim + 1)] + delta_u_delta_t * dt;
}


__global__ void advectVelocityYExplicitEuler(float* output_field, float* u_staggered, float* v_staggered, float* v_centered, float* v_field, int dim, float dt){
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
    output_field[IDX(j, i, dim)] = v_field[IDX(j, i, dim)] + delta_v_delta_t * dt;
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


void LaunchQuickVelocityYKernel(float* output_field, const int dimensions, const float timestep, const float* u, const float* v) {
    const int DIM = dimensions;
    const int BLOCK_DIM = 16;
    const int BLOCK_ROW_COUNT = ((DIM + 1) / BLOCK_DIM) + 1;
    const dim3 BLOCK(BLOCK_DIM, BLOCK_DIM, 1);
    const dim3 GRID(BLOCK_ROW_COUNT, BLOCK_ROW_COUNT, 1);

    // Setup Device Pointers for u and v
    float* d_u, * d_v;
    cudaMalloc(&d_u, (DIM + 1) * DIM * sizeof(float));
    cudaMalloc(&d_v, DIM * (DIM + 1) * sizeof(float));
    cudaMemcpy(d_u, u, (DIM + 1) * DIM * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, v, DIM * (DIM + 1) * sizeof(float), cudaMemcpyHostToDevice);

    float* d_staggered_u;
    cudaMalloc(&d_staggered_u, (DIM + 1) * (DIM + 1) * sizeof(float));
    interpolateStaggeredVelocityX CUDA_CALL(GRID, BLOCK) (d_staggered_u, d_u, DIM);

    float* d_staggered_v;
    cudaMalloc(&d_staggered_v, (DIM + 1) * (DIM + 1) * sizeof(float));
    upwindStaggeredVelocityQuickY CUDA_CALL(GRID, BLOCK) (d_staggered_v, d_staggered_u, d_v, DIM);

    float* d_centered_v;
    cudaMalloc(&d_centered_v, DIM * DIM * sizeof(float));
    upwindCenteredVelocityQuickY CUDA_CALL(GRID, BLOCK) (d_centered_v, d_v, DIM);

    float* d_out;
    cudaMalloc(&d_out, (DIM + 1) * DIM * sizeof(float));
    advectVelocityYExplicitEuler CUDA_CALL(GRID, BLOCK) (d_out, d_staggered_u, d_staggered_v, d_centered_v, d_v, DIM, timestep);
    cudaMemcpy(output_field, d_out, (DIM + 1) * DIM * sizeof(float), cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_u);
    cudaFree(d_v);
    cudaFree(d_staggered_u);
    cudaFree(d_staggered_v);
    cudaFree(d_centered_v);
    cudaFree(d_out);
}


void LaunchQuickVelocityXKernel(float* output_field, const int dimensions, const float timestep, const float* u, const float* v){
    const int DIM = dimensions;
    const int BLOCK_DIM = 16;
    const int BLOCK_ROW_COUNT = ((DIM + 1) / BLOCK_DIM) + 1;
    const dim3 BLOCK(BLOCK_DIM, BLOCK_DIM, 1);
    const dim3 GRID(BLOCK_ROW_COUNT, BLOCK_ROW_COUNT, 1);

    // Setup Device Pointers for u and v
    float* d_u, * d_v;
    cudaMalloc(&d_u, (DIM + 1) * DIM * sizeof(float));
    cudaMalloc(&d_v, DIM * (DIM + 1) * sizeof(float));
    cudaMemcpy(d_u, u, (DIM + 1) * DIM * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, v, DIM * (DIM + 1) * sizeof(float), cudaMemcpyHostToDevice);

    float* d_staggered_v;
    cudaMalloc(&d_staggered_v, (DIM + 1) * (DIM + 1) * sizeof(float));
    interpolateStaggeredVelocityY CUDA_CALL(GRID, BLOCK) (d_staggered_v, d_v, DIM);

    float* d_staggered_u;
    cudaMalloc(&d_staggered_u, (DIM + 1) * (DIM + 1) * sizeof(float));
    upwindStaggeredVelocityQuickX CUDA_CALL(GRID, BLOCK) (d_staggered_u, d_u, d_staggered_v, DIM);

    float* d_centered_u;
    cudaMalloc(&d_centered_u, DIM * DIM * sizeof(float));
    upwindCenteredVelocityQuickX CUDA_CALL(GRID, BLOCK) (d_centered_u, d_u, DIM);

    float* d_out;
    cudaMalloc(&d_out, (DIM + 1) * DIM * sizeof(float));
    advectVelocityXExplicitEuler CUDA_CALL(GRID, BLOCK) (d_out, d_staggered_u, d_staggered_v, d_centered_u, d_u, DIM, timestep);
    cudaMemcpy(output_field, d_out, (DIM + 1) * DIM * sizeof(float), cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_u);
    cudaFree(d_v);
    cudaFree(d_staggered_u);
    cudaFree(d_staggered_v);
    cudaFree(d_centered_u);
    cudaFree(d_out);
}


