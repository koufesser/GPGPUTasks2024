#ifdef __CLION_IDE__
    #include <libgpu/opencl/cl/clion_defines.cl>
#endif


#line 6

// TILE_SIZE и WORK_PER_THREAD задаются через поле 'defines' в кернел конфиге

__kernel void matrix_multiplication_naive(__global float *a, __global float *b, __global float *c, unsigned int M, unsigned int K, unsigned int N) {
    int i = get_global_id(0);
    int j = get_global_id(1);
    float sum = 0.0f;
    for (int k = 0; k < K; ++k) {
        sum += a[j * K + k] * b[k * N + i];
    }
    c[j * N + i] = sum;
}

#ifdef TILE_SIZE
__kernel void matrix_multiplication_local(__global float *a, __global float *b, __global float *c, unsigned int M, unsigned int K, unsigned int N) {
    int i = get_global_id(0);
    int j = get_global_id(1);
    int local_i = get_local_id(0);
    int local_j = get_local_id(1);

    __local float tileA[TILE_SIZE][TILE_SIZE];
    __local float tileB[TILE_SIZE][TILE_SIZE];

    float sum = 0.0f;
    for (int tileK = 0; tileK * TILE_SIZE < K; tileK++) {
        tileA[local_j][local_i] = a[j * K + local_i + tileK * TILE_SIZE];
        tileB[local_j][local_i] = b[(local_j + tileK * TILE_SIZE) * N + i];

        barrier(CLK_LOCAL_MEM_FENCE);

        for (int l = 0; l < TILE_SIZE; l++) {
            sum += tileA[local_j][l] * tileB[l][local_i];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }
    c[j * N + i] = sum;
   }
#endif

#if defined(TILE_SIZE) && defined(WORK_PER_THREAD)
__kernel void matrix_multiplication_local_wpt(__global float *a, __global float *b, __global float *c, unsigned int M, unsigned int K, unsigned int N) {

    int i = get_global_id(0);
    int j = get_global_id(1);
    int local_i = get_local_id(0);
    int local_j = get_local_id(1);

    __local float tileA[TILE_SIZE][TILE_SIZE];
    __local float tileB[TILE_SIZE][TILE_SIZE];

     float sum[WORK_PER_THREAD];
     for (int l = 0; l < WORK_PER_THREAD; l++) {
        sum[l] = 0;
     }

    for (int tileK = 0; tileK * TILE_SIZE < K; tileK++) {

        for (int thread = 0; thread < WORK_PER_THREAD; thread++) {
            tileA[local_j * WORK_PER_THREAD + thread][local_i] = a[(j * WORK_PER_THREAD + thread) * K + local_i + tileK * TILE_SIZE];
            tileB[local_j * WORK_PER_THREAD + thread][local_i] = b[(local_j * WORK_PER_THREAD + thread + tileK * TILE_SIZE) * N + i];
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        for (int l = 0; l < TILE_SIZE; l++) {
            for (int thread = 0; thread < WORK_PER_THREAD; thread++) {
                sum[thread] += tileA[local_j * WORK_PER_THREAD + thread][l] * tileB[l][local_i];
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    for (int thread = 0; thread < WORK_PER_THREAD; thread++) {
        c[(j * WORK_PER_THREAD + thread) * N + i] = sum[thread];
    }

}
#endif
