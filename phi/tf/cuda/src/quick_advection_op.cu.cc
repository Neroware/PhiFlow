#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <numeric>


void LaunchQuickKernel(int* testin) {
	*testin = 42;	
}
