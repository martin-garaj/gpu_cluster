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


// header
#include "Cuda_kernel.cuh"


extern "C"
{
	/** Function to be called from a C++ object when the kernel<<< , >>> is to be executed
	 * @param number of blocks to be used on GPU to execute the kernel
	 * @param number of threads to be used on GPU to execute the kernel
	 * @param object representing separate input data for every GPU element
	 * @param object representing separate output data for every GPU element
	 * @param object representing shared input data for all GPU elements
	 * @return _SUCCESS_
	 */
	__host__ int kernel_execute(int blocks, int threads, GPU_grid_in &mem_grid_in, GPU_grid_out &mem_grid_out, GPU_shared_in &mem_shared_in){
		// this is a relay function, the actuall kernel is defined in a separate file
		kernel_by_ref<<<blocks, threads>>>(mem_grid_in, mem_grid_out, mem_shared_in); // has to return void
		return _SUCCESS_;
	};


	/** Delays the execution of the CPU until the GPU is done, this is to prevent simultaneous acceess to CPU-GPU shared memory
	 * @return _SUCCESS_
	 */ 
	__host__ int kernel_synchronize(void){
		cudaDeviceSynchronize(); // prevent CPU and GPU asynchronous execution
		return _SUCCESS_;
	};



}
