#include <stdio.h>
#include <stdlib.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

const int allocation_size = 1 * 1024 * 1024 * 1024;

void* cpu_p;
void* gpu_p;

void cpu_alloc(){
    cpu_p = malloc(allocation_size);
}

int main() {
    cpu_alloc();
    int rc = getchar();
}