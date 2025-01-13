#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>
#include <libutils/fast_random.h>
#include <libutils/misc.h>
#include <libutils/timer.h>

#include "cl/sum_cl.h"
#include <cassert>

template<typename T>
void raiseFail(const T &a, const T &b, std::string message, std::string filename, int line) {
    if (a != b) {
        std::cerr << message << " But " << a << " != " << b << ", " << filename << ":" << line << std::endl;
        throw std::runtime_error(message);
    }
}

#define EXPECT_THE_SAME(a, b, message) raiseFail(a, b, message, __FILE__, __LINE__)

void run(gpu::Device &device, const std::string &kernelName, const unsigned int n, const unsigned int reference_sum,
         const unsigned int benchmarkingIters, const std::vector<unsigned int> &as, gpu::WorkSize ws) {

    gpu::Context context;
    context.init(device.device_id_opencl);
    context.activate();
    {
        gpu::gpu_mem_32u as_gpu;
        gpu::gpu_mem_32u sum_gpu;

        as_gpu.resizeN(as.size());
        sum_gpu.resizeN(1);
        assert(n == as.size());
        as_gpu.writeN(as.data(), as.size());
        ocl::Kernel kernel(sum_kernel, sum_kernel_length, kernelName);
        bool printLog = false;
        kernel.compile(printLog);
        unsigned int sum[1];
        timer t;
        for (int i = 0; i < benchmarkingIters; ++i) {
            sum[0] = 0;
            sum_gpu.writeN(&sum[0],1);
            kernel.exec(ws,
                as_gpu, sum_gpu, n);
            sum_gpu.readN(sum, 1);
            EXPECT_THE_SAME(reference_sum, sum[0], "GPU result should be consistent!");
            t.nextLap();
        }
        std::cout << kernelName << std::endl;
        std::cout << "GPU:     " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "GPU:     " << (n / 1000.0 / 1000.0) / t.lapAvg() << " millions/s" << std::endl;
    }

}

int main(int argc, char **argv) {
    int benchmarkingIters = 10;

    unsigned int reference_sum = 0;
    const unsigned int n = 100 * 1000 * 1000;
    std::vector<unsigned int> as(n, 0);
    FastRandom r(42);
    for (int i = 0; i < n; ++i) {
        as[i] = (unsigned int) r.next(0, std::numeric_limits<unsigned int>::max() / n);
        reference_sum += as[i];
    }

    {
        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            unsigned int sum = 0;
            for (int i = 0; i < n; ++i) {
                sum += as[i];
            }
            EXPECT_THE_SAME(reference_sum, sum, "CPU result should be consistent!");
            t.nextLap();
        }
        std::cout << "CPU:     " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "CPU:     " << (n / 1000.0 / 1000.0) / t.lapAvg() << " millions/s" << std::endl;
    }

    {
        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            unsigned int sum = 0;
#pragma omp parallel for reduction(+ : sum)
            for (int i = 0; i < n; ++i) {
                sum += as[i];
            }
            EXPECT_THE_SAME(reference_sum, sum, "CPU OpenMP result should be consistent!");
            t.nextLap();
        }
        std::cout << "CPU OMP: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "CPU OMP: " << (n / 1000.0 / 1000.0) / t.lapAvg() << " millions/s" << std::endl;
    }

    {
        const unsigned int workGroupSize = 64;
        const unsigned int valuesPerWorkItem = 32;
        const unsigned int workItems = (n + workGroupSize - 1) / workGroupSize * workGroupSize;
        // TODO: implement on OpenCL
        gpu::Device device = gpu::chooseGPUDevice(argc, argv);
        run(device, "atomic_sum", n, reference_sum, benchmarkingIters, as, gpu::WorkSize(workGroupSize, workItems));
        run(device, "cycle_sum", n, reference_sum, benchmarkingIters, as, gpu::WorkSize(workGroupSize, workItems / valuesPerWorkItem));
        run(device, "cycle_coalesced_sum", n, reference_sum, benchmarkingIters, as, gpu::WorkSize(workGroupSize, workItems / valuesPerWorkItem));
        run(device, "local_mem_sum", n, reference_sum, benchmarkingIters, as, gpu::WorkSize(workGroupSize, workItems));
        run(device, "tree_sum", n, reference_sum, benchmarkingIters, as, gpu::WorkSize(workGroupSize, workItems));
    }
}
