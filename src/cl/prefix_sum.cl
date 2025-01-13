#ifdef __CLION_IDE__
    #include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

__kernel void prefix_sum_binary(__global unsigned int* sum, unsigned int n, unsigned int rate)
{
       const int i = (get_global_id(0) + 1) * rate - 1;
       if (i < n)
            sum[i] += sum[i - rate / 2];
}

__kernel void prefix_sum_second_part(__global unsigned int* sum, unsigned int n, unsigned int rate)
{
    const int i = (get_global_id(0) + 1) * rate - 1 + rate / 2;
    if (i < n)
        sum[i] += sum[i - rate / 2];
}