#ifdef __CLION_IDE__
#include "clion_defines.cl"
#endif

#line 5


unsigned int lower_bound(__global const int *array, unsigned int len, int value)
{
    unsigned int r = len;
    unsigned int l = 0;

    while (l < r) {
        int m = (l + r) / 2;
        if (array[m] < value) {
            l = m + 1;
        } else {
            r = m;
        }
    }
    return l;
}

unsigned int upper_bound(__global const int *array, unsigned int len, int value)
{
    unsigned int r = len;
    unsigned int l = 0;

    while (l < r) {
        int m = (l + r) / 2;
        if (array[m] <= value) {
            l = m + 1;
        } else {
            r = m;
        }
    }
    return l;
}

__kernel void merge_global(__global const int *as, __global int *bs, unsigned int block_size)
{
    const int i = get_global_id(0);
    const int value = as[i];

    const int double_block = block_size * 2;
    const unsigned int block_index = i % block_size;
    const unsigned int block_start = i - block_index;
    const unsigned int double_block_index = i % double_block ;
    const unsigned int double_block_start = i - double_block_index;
    unsigned int other_block_index = 0;

    __global const int *double_block_start_pointer = as + double_block_start;

    if (double_block_index < block_size ) {
        __global const int *other = double_block_start_pointer + block_size;
        other_block_index = lower_bound(other, block_size, value);
    } else {
        __global const int *other = double_block_start_pointer;
        other_block_index = upper_bound(other, block_size, value);
    }

    bs[double_block_start + block_index + other_block_index] = value;
    return;
}

__kernel void calculate_indices(__global const int *as, __global unsigned int *inds, unsigned int block_size)
{

}

__kernel void merge_local(__global const int *as, __global const unsigned int *inds, __global int *bs, unsigned int block_size)
{

}
