#ifdef __CLION_IDE__
    #include <libgpu/opencl/cl/clion_defines.cl>
#endif

#define TILE_SIZE 16
#line 6

__kernel void matrix_transpose_naive( __global float *a, __global float *at, unsigned int h, unsigned int w)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    if (i >= w)
        return;
    if (j >= h)
        return;
    float x = a[j * w + i];
    at[i * h + j] = x;
}


__kernel void matrix_transpose_local_bad_banks( __global float *a, __global float *at, unsigned int h, unsigned int w)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    __local float tile[TILE_SIZE][TILE_SIZE];
    int local_i = get_local_id(0);
    int local_j = get_local_id(1);


    int start_i = i - local_i;
    int start_j = j - local_j;
    int transposed_i = start_j + local_i;
    int transposed_j = start_i + local_j;

    tile[local_j][local_i] = a[j * w + i];
    barrier(CLK_LOCAL_MEM_FENCE);
    at[transposed_j * w + transposed_i] = tile[local_i][local_j];
}

__kernel void matrix_transpose_local_good_banks( __global float *a, __global float *at, unsigned int h, unsigned int w)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    __local float tile[TILE_SIZE][TILE_SIZE + 1];
    int local_i = get_local_id(0);
    int local_j = get_local_id(1);

    int start_i = i - local_i;
    int start_j = j - local_j;
    int transposed_i = start_j + local_i;
    int transposed_j = start_i + local_j;

    int indx = local_j * TILE_SIZE + local_i;
    int biased_j = indx / (TILE_SIZE + 1);
    int biased_i = indx % (TILE_SIZE + 1);

    tile[biased_j][biased_i] = a[j*w + i];
    barrier(CLK_LOCAL_MEM_FENCE);

    indx = local_i * TILE_SIZE + local_j;
    biased_j = indx / (TILE_SIZE + 1);
    biased_i = indx % (TILE_SIZE + 1);
    at[transposed_j * w + transposed_i] = tile[biased_j][biased_i];
}
