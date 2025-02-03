#ifndef GPU_H
#define GPU_H

#ifdef __CUDACC__
#define GPUd() __host__ __device__
#else
#define GPUd()
#endif

#endif  // GPU_H