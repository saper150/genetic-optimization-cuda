#include "./Utils.cuh"

__device__ ArrayWrapper twoBiggestValues(int *arr, int size) {

    ArrayValue a;
    ArrayValue b;

    for (size_t i = 0; i < size; i++) {
        if (arr[i] >= a.val) {
            b = a;
            a.index = i;
            a.val = arr[i];
        }
    }
    return {{a, b}};
}
