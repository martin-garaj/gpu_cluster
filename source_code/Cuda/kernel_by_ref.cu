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

	/** This function is the actuall kernel executed at the GPU 
	 * @param object representing separate input data for every GPU element
	 * @param object representing separate output data for every GPU element
	 * @param object representing shared input data for all GPU elements
	 */
	__global__ void kernel_by_ref(GPU_grid_in &mem_grid_in, GPU_grid_out &mem_grid_out, GPU_shared_in &mem_shared_in){
		//--------------------------------KERNEL_INITIALIZATION--------------------------------//
		// identify kernel
		int id = blockIdx.x * blockDim.x + threadIdx.x;

		// load data associated with the kernel
		elements grid_element = mem_grid_in.element[id];

		// temporary variables
		float t_result;

		//--------------------------------KERNEL_EXECUTION--------------------------------//
		// loop through layers
		for(int x = 0; x<NN_layers; x++){
			// loop through neurons
			for(int y = 0; y < grid_element.NN.layout[x]; y++){
				// loop through weights
				for(int z = 0; z< grid_element.NN.layout[x]; z++){
					t_result = t_result + grid_element.NN.weight[x][y][z];
				}
			}
		}

		t_result = t_result + mem_shared_in.debug;

		//--------------------------------KERNEL_FINALIZATION--------------------------------//
		// write the result to GPU memory
		mem_grid_out.result[id] = t_result;
	};

}
