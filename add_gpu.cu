
#include <cstdio>
#include <cstdint>
#include <Python.h>
#include "dlpack.h"

__global__
void kernel_add(float* a, size_t length, float value) {
    size_t idx = threadIdx.x + blockDim.x*blockIdx.x;
    if (idx < length) {
        a[idx] += idx + value;
    }
}

void print_info(const DLTensor* tensor)
{
    printf("context : %d %d\n", tensor->ctx.device_type, tensor->ctx.device_id);
    printf("dtype : %d %d %d\n", tensor->dtype.code, tensor->dtype.bits, tensor->dtype.lanes);
    printf("ndim : %d\n", tensor->ndim);
    printf("shape : ");
    size_t len = 1;
    for (int i=0; i<tensor->ndim; ++i) {
        len *= tensor->shape[i];
        printf("%ld ", tensor->shape[i]);
    }
    printf("\n");
    printf("strides : ");
    for (int i=0; i<tensor->ndim; ++i) {
        printf("%ld ", tensor->strides[i]);
    }
    printf("\n");
    printf("byte_offset : %lu\n", tensor->byte_offset);
}

void cadd(void* obj, float value)
{
//   printf("%p %d\n", obj, value);
    DLTensor* tensor = (DLTensor*)obj;
    // print_info(tensor);
    size_t len = 1;
    for (int i=0; i<tensor->ndim; ++i) {
        len *= tensor->shape[i];
    }
    float* data = (float*)tensor->data;
    kernel_add<<<(len+127)/128, 128>>>(data, len, value);
}
