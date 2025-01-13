__kernel void bitonic(__global int* array, unsigned int block_size, unsigned int small_block_size)
{
    const int i = get_global_id(0);
    const int small_block_id = i / small_block_size;
    const int real_id = small_block_id * small_block_size * 2 + i % small_block_size;

    int is_growing = 1;

    const int block_id = i / block_size;
    if (block_id % 2 != 0) {
        is_growing = -1;
    }

    if (is_growing * array[real_id] > is_growing * array[real_id + small_block_size]) {
        const int t = array[real_id + small_block_size];
        array[real_id + small_block_size] = array[real_id];
        array[real_id] = t;
    }
}