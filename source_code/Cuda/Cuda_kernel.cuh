/* CREDITS
 *   Author  : Martin Garaj <garaj.martin@gmail.com>
 *   Date    : 12/2017
 *   Project : HPC MPI Testing application
 *
 * REDISTRIBUTION
 *   The software is meant for internal use at City University of Hong Kong only.
 *   Further redistribution/sharing of the software or any parts of the source code without author's permission is disapproved.
 *
 * DISCLAIMER
 *   This software comes as is, without any warranty. Use on your own risk.
 *
 * CHANGE LOG
 *   Please log any changes below, log the Author, Date and Change done.
 *        Author     |     Date     |   Change
 *                   |  YYYY/MM/DD  |   
 */




#ifndef CUDA_KERNEL_CUH_
#define CUDA_KERNEL_CUH_

// Data objects
#include "GPU_grid_in.cuh"
#include "GPU_grid_out.cuh"
#include "GPU_shared_in.cuh"

// return values
#include "return_values.h"

// Why not to define Kernel_object, the same way as the Cuda_GPU ?
// This is due to C++ mangling, which changes the names of the functions, 
// due to overloading. While mangling is extremely usefull, the NVCC (or Linker?) 
// is not accustomed to handle the "mangled" names produced by C++ compiler.
// Therefore the functions compiled by C++ compiler, need to keep their name intact,
// this is done by using 'extern "C"', which forces the compiler to compile the functions
// as C functions. Since C does not know overloading feature, the names are not mangled,
// so the functions can be linked by the Linker(?).
//
// This affects only the kernel function, which has to be defined as global, 
// therefore shared between CPU and GPU compiler/linker.

extern "C"
{
		__host__ int kernel_execute(int blocks, int threads, GPU_grid_in &mem_grid_in, GPU_grid_out &mem_grid_out, GPU_shared_in &mem_shared_in);

		__host__ int kernel_synchronize(void);

		__global__ void kernel_by_ref(GPU_grid_in &mem_grid_in, GPU_grid_out &mem_grid_out, GPU_shared_in &mem_shared_in);
}




#endif /* CUDA_KERNEL_CUH_ */
