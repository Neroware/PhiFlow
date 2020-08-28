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
__device__ float* coefficients(float vel1, float vel2) {
    float* c = (float*) malloc(5 * sizeof(float));
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

    return c;
}


__global__ void gradientFieldQuick(float* output_field, float* rho, float* u, float* v, float* grad, int dim, int padding, float dt) {
    int i = CUDA_THREAD_COL;
    int j = CUDA_THREAD_ROW;

    if (i >= dim || j >= dim) {
        return;
    }

    float u1, u2, u3, u4, u5, u6;
    u1 = u[pidx(j, i - 2, dim + 1, padding)];
    u2 = u[pidx(j, i - 1, dim + 1, padding)];
    u3 = u[pidx(j, i, dim + 1, padding)];
    u4 = u[pidx(j, i + 1, dim + 1, padding)];
    u5 = u[pidx(j, i + 2, dim + 1, padding)];
    u6 = u[pidx(j, i + 3, dim + 1, padding)];
    float* cs_u1 = coefficients(u1, u2);
    float* cs_u2 = coefficients(u2, u3);
    float* cs_u3 = coefficients(u3, u4);
    float* cs_u4 = coefficients(u4, u5);
    float* cs_u5 = coefficients(u5, u6);

    float v1, v2, v3, v4, v5, v6;
    v1 = v[pidx(j - 2, i, dim, padding)];
    v2 = v[pidx(j - 1, i, dim, padding)];
    v3 = v[pidx(j, i, dim, padding)];
    v4 = v[pidx(j + 1, i, dim, padding)];
    v5 = v[pidx(j + 2, i, dim, padding)];
    v6 = v[pidx(j + 3, i, dim, padding)];
    float* cs_v1 = coefficients(v1, v2);
    float* cs_v2 = coefficients(v2, v3);
    float* cs_v3 = coefficients(v3, v4);
    float* cs_v4 = coefficients(v4, v5);
    float* cs_v5 = coefficients(v5, v6);

    float g[9];
    for (int k = 0; k < 9; k++) {
        g[k] = 0.0f;
    }
    g[2] = grad[IDX(j, i, dim)];
    if (i > 1)
        g[0] = grad[IDX(j, i - 2, dim)];
    if (i > 0)
        g[1] = grad[IDX(j, i - 1, dim)];
    if(i < dim - 1)
        g[3] = grad[IDX(j, i + 1, dim)];
    if (i < dim - 2)
        g[4] = grad[IDX(j, i + 2, dim)];
    if (j > 1)
        g[5] = grad[IDX(j - 2, i, dim)];
    if (j > 0)
        g[6] = grad[IDX(j - 1, i, dim)];
    if (j < dim - 1)
        g[7] = grad[IDX(j + 1, i, dim)];
    if (j < dim - 2)
        g[8] = grad[IDX(j + 2, i, dim)];

    output_field[pidx(j, i, dim, padding)] = g[2] * -(cs_u3[2] * dt + cs_v3[2] * dt) +
        g[0] * -(cs_u1[4] * dt) +
        g[1] * -(cs_u2[3] * dt) +
        g[3] * -(cs_u4[1] * dt) +
        g[4] * -(cs_u5[0] * dt) +
        g[5] * -(cs_v1[4] * dt) +
        g[6] * -(cs_v2[3] * dt) +
        g[7] * -(cs_v4[1] * dt) +
        g[8] * -(cs_v5[0] * dt);

    free(cs_u1);
    free(cs_u2);
    free(cs_u3);
    free(cs_u4);
    free(cs_u5);
    free(cs_v1);
    free(cs_v2);
    free(cs_v3);
    free(cs_v4);
    free(cs_v5);
}


__global__ void gradientVelocityXQuick(float* output_field, float* rho, float* u, float* v, float* grad, int dim, int padding, float dt) {
    int i = CUDA_THREAD_COL;
    int j = CUDA_THREAD_ROW;

    if (i >= dim + 1 || j >= dim) {
        return;
    }

    float u1, u2, u3;
    u1 = u[pidx(j, i - 1, dim + 1, padding)];
    u2 = u[pidx(j, i, dim + 1, padding)];
    u3 = u[pidx(j, i + 1, dim + 1, padding)];

    float* cs_u1 = coefficients(u1, u2);
    float* cs_u2 = coefficients(u1, u2 + 1.0f);
    float* cs_u3 = coefficients(u2, u3);
    float* cs_u4 = coefficients(u2 + 1.0f, u3);

    output_field[pidx(j, i, dim + 1, padding)] = 0.0f;
    float diff_1, diff_2;
    diff_1 = diff_2 = 0.0f;
    float grad_1, grad_2;
    grad_1 = grad_2 = 0.0f;

    if (i > 0) {
        float delta_u_rho_delta_x_1 =
            cs_u1[0] * rho[pidx(j, i - 3, dim, padding)] +
            cs_u1[1] * rho[pidx(j, i - 2, dim, padding)] +
            cs_u1[2] * rho[pidx(j, i - 1, dim, padding)] +
            cs_u1[3] * rho[pidx(j, i, dim, padding)] +
            cs_u1[4] * rho[pidx(j, i + 1, dim, padding)];
        float delta_u_rho_delta_x_2 =
            cs_u2[0] * rho[pidx(j, i - 3, dim, padding)] +
            cs_u2[1] * rho[pidx(j, i - 2, dim, padding)] +
            cs_u2[2] * rho[pidx(j, i - 1, dim, padding)] +
            cs_u2[3] * rho[pidx(j, i, dim, padding)] +
            cs_u2[4] * rho[pidx(j, i + 1, dim, padding)];
        diff_1 = (-delta_u_rho_delta_x_2) - (-delta_u_rho_delta_x_1);
        grad_1 = grad[IDX(j, i - 1, dim)];
        output_field[pidx(j, i, dim + 1, padding)] = diff_1;
    }
    if (i < dim - 1) {
        float delta_u_rho_delta_x_1 =
            cs_u3[0] * rho[pidx(j, i - 2, dim, padding)] +
            cs_u3[1] * rho[pidx(j, i - 1, dim, padding)] +
            cs_u3[2] * rho[pidx(j, i, dim, padding)] +
            cs_u3[3] * rho[pidx(j, i + 1, dim, padding)] +
            cs_u3[4] * rho[pidx(j, i + 2, dim, padding)];
        float delta_u_rho_delta_x_2 =
            cs_u4[0] * rho[pidx(j, i - 2, dim, padding)] +
            cs_u4[1] * rho[pidx(j, i - 1, dim, padding)] +
            cs_u4[2] * rho[pidx(j, i, dim, padding)] +
            cs_u4[3] * rho[pidx(j, i + 1, dim, padding)] +
            cs_u4[4] * rho[pidx(j, i + 2, dim, padding)];
        diff_2 = (-delta_u_rho_delta_x_2) - (-delta_u_rho_delta_x_1);
        grad_2 = grad[IDX(j, i, dim)];
    }

    //output_field[pidx(j, i, dim + 1, padding)] = diff_1 + diff_2;

    free(cs_u1);
    free(cs_u2);
    free(cs_u3);
    free(cs_u4);
}


__global__ void gradientVelocityYQuick(float* output_field, float* rho, float* u, float* v, float* grad, int dim, int padding, float dt) {
    int i = CUDA_THREAD_COL;
    int j = CUDA_THREAD_ROW;

    if (i >= dim || j >= dim + 1) {
        return;
    }

    float v1, v2, v3;
    v1 = v[pidx(j - 1, i, dim, padding)];
    v2 = v[pidx(j, i, dim, padding)];
    v3 = v[pidx(j + 1, i, dim, padding)];

    float* cs_v1 = coefficients(v1, v2);
    float* cs_v2 = coefficients(v1, v2 + 1.0f);
    float* cs_v3 = coefficients(v2, v3);
    float* cs_v4 = coefficients(v2 + 1.0f, v3);

    output_field[pidx(j, i, dim, padding)] = 0.0f;
    if (j > 0) {
        float delta_v_rho_delta_y_1 =
            cs_v1[0] * rho[pidx(j - 3, i, dim, padding)] +
            cs_v1[1] * rho[pidx(j - 2, i, dim, padding)] +
            cs_v1[2] * rho[pidx(j - 1, i, dim, padding)] +
            cs_v1[3] * rho[pidx(j, i, dim, padding)] +
            cs_v1[4] * rho[pidx(j + 1, i, dim, padding)];
        float delta_v_rho_delta_y_2 =
            cs_v2[0] * rho[pidx(j - 3, i, dim, padding)] +
            cs_v2[1] * rho[pidx(j - 2, i, dim, padding)] +
            cs_v2[2] * rho[pidx(j - 1, i, dim, padding)] +
            cs_v2[3] * rho[pidx(j, i, dim, padding)] +
            cs_v2[4] * rho[pidx(j + 1, i, dim, padding)];
        float diff_1 = (-delta_v_rho_delta_y_2) - (-delta_v_rho_delta_y_1);
        output_field[pidx(j, i, dim, padding)] += diff_1 * grad[IDX(j - 1, i, dim)] * dt;
    }
    if (j < dim - 1) {
        float delta_v_rho_delta_y_1 =
            cs_v3[0] * rho[pidx(j - 2, i, dim, padding)] +
            cs_v3[1] * rho[pidx(j - 1, i, dim, padding)] +
            cs_v3[2] * rho[pidx(j, i, dim, padding)] +
            cs_v3[3] * rho[pidx(j + 1, i, dim, padding)] +
            cs_v3[4] * rho[pidx(j + 2, i, dim, padding)];
        float delta_v_rho_delta_y_2 =
            cs_v4[0] * rho[pidx(j - 2, i, dim, padding)] +
            cs_v4[1] * rho[pidx(j - 1, i, dim, padding)] +
            cs_v4[2] * rho[pidx(j, i, dim, padding)] +
            cs_v4[3] * rho[pidx(j + 1, i, dim, padding)] +
            cs_v4[4] * rho[pidx(j + 2, i, dim, padding)];
        float diff_2 = (-delta_v_rho_delta_y_2) - (-delta_v_rho_delta_y_1);
        output_field[pidx(j, i, dim, padding)] += diff_2 * grad[IDX(j, i, dim)] * dt;
    }

    free(cs_v1);
    free(cs_v2);
    free(cs_v3);
    free(cs_v4);
}




/*
__device__ float* coefficients_derivative(float vel1, float vel2) {
    float* c = (float*) malloc(5 * sizeof(float));
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

    return c;
}


__global__ void gradientFieldQuick(float* output_field, float* rho, float* u, float* v, float* grad, int dim, int padding, float dt) {
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

    float* cs_u = coefficients(u1, u2);
    float* cs_v = coefficients(v1, v2);

    output_field[pidx(j, i, dim, padding)] = dt * -(cs_u[0] + cs_u[1] + cs_u[2] + cs_u[3] + cs_u[4]) - (cs_v[0] + cs_v[1] + cs_v[2] + cs_v[3] + cs_v[4]) * loss[IDX(j, i, dim)];
    
    free(cs_u);
    free(cs_v);
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
    float* cs_u = coefficients_derivative(u1, u2);
    
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

   free(cs_u);
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
    float* cs_v = coefficients_derivative(v1, v2);

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

    free(cs_v);
}
*/




void LaunchQUICKAdvectionScalarGradientKernel(float* output_grds, float* vel_u_grds, float* vel_v_grds, const int dimensions, const int padding, const float timestep, const float* rho, const float* u, const float* v, const float* grad){
    const int DIM = dimensions;
    const int PADDING = padding;
    const float DT = timestep;
    const int BLOCK_DIM = 16;
    const int BLOCK_ROW_COUNT = ((DIM + 1) / BLOCK_DIM) + 1;
    const dim3 BLOCK(BLOCK_DIM, BLOCK_DIM, 1);
    const dim3 GRID(BLOCK_ROW_COUNT, BLOCK_ROW_COUNT, 1);

    // Setup Device Pointers for u, v and rho
    float *d_rho, *d_u, *d_v, *d_grad;
    cudaMalloc(&d_rho, (DIM + 2 * PADDING) * (DIM + 2 * PADDING) * sizeof(float));
    cudaMalloc(&d_u, (DIM + 2 * PADDING + 1) * (DIM + 2 * PADDING) * sizeof(float));
    cudaMalloc(&d_v, (DIM + 2 * PADDING) * (DIM + 2 * PADDING + 1) * sizeof(float));
    cudaMalloc(&d_grad, DIM * DIM * sizeof(float));
    cudaMemcpy(d_rho, rho, (DIM + 2 * PADDING) * (DIM + 2 * PADDING) * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_u, u, (DIM + 2 * PADDING + 1) * (DIM + 2 * PADDING) * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, v, (DIM + 2 * PADDING) * (DIM + 2 * PADDING + 1) * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_grad, grad, DIM * DIM * sizeof(float), cudaMemcpyHostToDevice);

    // Setup output pointer
    float *d_out_field, *d_out_u, *d_out_v;
    cudaMalloc(&d_out_field, (DIM + 2 * PADDING) * (DIM + 2 * PADDING) * sizeof(float));
    cudaMalloc(&d_out_u, (DIM + 2 * PADDING + 1) * (DIM + 2 * PADDING) * sizeof(float));
    cudaMalloc(&d_out_v, (DIM + 2 * PADDING) * (DIM + 2 * PADDING + 1) * sizeof(float));

    gradientFieldQuick CUDA_CALL(GRID, BLOCK) (d_out_field, d_rho, d_u, d_v, d_grad, DIM, PADDING, DT);
    cudaMemcpy(output_grds, d_out_field, (DIM + 2 * PADDING) * (DIM + 2 * PADDING) * sizeof(float), cudaMemcpyDeviceToHost);

    gradientVelocityXQuick CUDA_CALL(GRID, BLOCK) (d_out_u, d_rho, d_u, d_v, d_grad, DIM, PADDING, DT);
    cudaMemcpy(vel_u_grds, d_out_u, (DIM + 2 * PADDING + 1) * (DIM + 2 * PADDING) * sizeof(float), cudaMemcpyDeviceToHost);

    gradientVelocityYQuick CUDA_CALL(GRID, BLOCK) (d_out_v, d_rho, d_u, d_v, d_grad, DIM, PADDING, DT);
    cudaMemcpy(vel_v_grds, d_out_v, (DIM + 2 * PADDING) * (DIM + 2 * PADDING + 1) * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_rho);
    cudaFree(d_u);
    cudaFree(d_v);
    cudaFree(d_grad);
    cudaFree(d_out_field);
    cudaFree(d_out_u);
    cudaFree(d_out_v);
}
