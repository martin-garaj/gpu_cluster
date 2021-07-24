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



#ifndef CUDA_GPU_HPP_
#define CUDA_GPU_HPP_

// return values
#include "return_values.h"

// std::size_t
#include <cstddef>

/** File Descripteor structure of GPU
 */
struct struct_fd_gpu{

	int clockRate;
	int integrated;
	int isMultiGpuBoard;
	int major;
	int managedMemory;
	int minor;
	int multiGpuBoardGroupID;
	int multiProcessorCount;
	char name[256];
	int pciBusID;
	int pciDeviceID;
	int pciDomainID;

	std::size_t totalConstMem; // should be size_t
	std::size_t totalGlobalMem; // should be size_t
};


/** Object representing a GPU
 */
class Cuda_GPU{
public:
	Cuda_GPU(void);
	~Cuda_GPU(void);

	int dev_count(void);
	int query(int gpu_id);

	int assign(int gpu_id);

	int get_fd_gpu(struct_fd_gpu * fd_gpu);

	int get_name(char * pt_name, int len);
	int get_pciBusID(void);
	int get_pciDeviceID(void);
	int get_pciDomainID(void);
	int get_major(void);
	int get_minor(void);
	int get_gpu_id(void);

private:
	int device_count;
	int assigned_gpu_id;
	struct_fd_gpu fd_gpu;
};

#endif /* CUDA_GPU_HPP_ */
