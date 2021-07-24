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
#include "Cuda_GPU.cuh"

/** Constructor
 */
Cuda_GPU::Cuda_GPU(void){
	// count visible GPUs
	int devCount;
	cudaGetDeviceCount(&devCount);

	// set private variable
	this->device_count = devCount;

	this->assigned_gpu_id = _ERROR_;

	// initialize file descriptor structure of gpu
	this->fd_gpu.clockRate = -1;
	this->fd_gpu.integrated = -1;
	this->fd_gpu.isMultiGpuBoard = -1;
	this->fd_gpu.major = -1;
	this->fd_gpu.managedMemory = -1;
	this->fd_gpu.minor = -1;
	this->fd_gpu.multiGpuBoardGroupID = -1;
	this->fd_gpu.multiProcessorCount = -1;
	strncpy(this->fd_gpu.name , "NOT_ASSIGNED", 250);
	this->fd_gpu.pciBusID = -1;
	this->fd_gpu.pciDeviceID = -1;
	this->fd_gpu.pciDomainID = -1;
	this->fd_gpu.totalConstMem = 0;
	this->fd_gpu.totalGlobalMem = 0;

};

/** Destructor
 */
Cuda_GPU::~Cuda_GPU(void){};


/** Returns the number of visible cuda-capable GPUs
 * @return number of visible GPUs
 */
int Cuda_GPU::dev_count(void){
	return this->device_count;
};


/** Assigns a GPU according to provided GPU ID
 * @param GPU ID ranging from 0 to (cudaGetDeviceCount() - 1)
 * @return _SUCCESS_ / _ERROR_
 */
int Cuda_GPU::assign(int gpu_id){

	if( (gpu_id > this->device_count-1) or (gpu_id < 0) ){
		return _ERROR_;
	}else if( cudaSuccess == cudaSetDevice(gpu_id) ){
		// remember the assigned gpu id
		this->assigned_gpu_id = gpu_id;
		// store the data about the assigned gpu
		query(gpu_id);
		return _SUCCESS_;
	}

	return _ERROR_;
};


/** Fills in the private GPU File Descriptor structure
 * @param GPU ID ranging from 0 to (cudaGetDeviceCount() - 1)
 * @return _SUCCESS_ / _ERROR_
 */
int Cuda_GPU::query(int gpu_id){

	if( (gpu_id > this->device_count-1) or (gpu_id < 0)){
		return _ERROR_;
	}else{
		// use Cuda to get GPU information
		cudaDeviceProp devProp;
		cudaGetDeviceProperties(&devProp, gpu_id);

		this->fd_gpu.clockRate = devProp.clockRate;
		this->fd_gpu.integrated = devProp.integrated;
		this->fd_gpu.isMultiGpuBoard = devProp.isMultiGpuBoard;
		this->fd_gpu.major = devProp.major;
		this->fd_gpu.managedMemory = devProp.managedMemory;
		this->fd_gpu.minor = devProp.minor;
		this->fd_gpu.multiGpuBoardGroupID = devProp.multiGpuBoardGroupID;
		this->fd_gpu.multiProcessorCount = devProp.multiProcessorCount;
		strncpy(this->fd_gpu.name, devProp.name, 256);
		this->fd_gpu.pciBusID = devProp.pciBusID;
		this->fd_gpu.pciDeviceID = devProp.pciDeviceID;
		this->fd_gpu.pciDomainID = devProp.pciDomainID;
		this->fd_gpu.totalConstMem = devProp.totalConstMem;
		this->fd_gpu.totalGlobalMem = devProp.totalGlobalMem;

	}

	return _SUCCESS_;
};

/** Copies the content of private GPU File Descriptor into provided structure
 * @param struct_fd_gpu structure defined in Cuda_GPU::
 * @return _SUCCESS_ / _ERROR_
 */
int Cuda_GPU::get_fd_gpu(struct_fd_gpu * pt_fd_gpu){
	if(this->assigned_gpu_id != -1){
		*pt_fd_gpu = this->fd_gpu;
		return _SUCCESS_;
	}
	return _ERROR_;
};

/** Getter of the GPU name
 * @param pointer to char array
 * @param lenght to be copied
 * @return _SUCCESS_ / _ERROR_
 */
int Cuda_GPU::get_name(char * pt_name, int len){

	if(this->assigned_gpu_id != _ERROR_){
		strncpy(pt_name, this->fd_gpu.name, len);
		return _SUCCESS_;
	}
	return _ERROR_;
};

/** Getter of the PCI Bus ID
 * @return PCI Bus ID / _ERROR_
 */
int Cuda_GPU::get_pciBusID(void){
	if(this->assigned_gpu_id != _ERROR_){
		return this->fd_gpu.pciBusID;
	}
	return _ERROR_;
};

/** Getter of the PCI Device ID
 * @return PCI Device ID / _ERROR_
 */
int Cuda_GPU::get_pciDeviceID(void){
	if(this->assigned_gpu_id != _ERROR_){
		return this->fd_gpu.pciDeviceID;
	}
	return _ERROR_;
};

/** Getter of the PCI Domain ID
 * @return PCI Domain ID / _ERROR_
 */
int Cuda_GPU::get_pciDomainID(void){
	if(this->assigned_gpu_id != _ERROR_){
		return this->fd_gpu.pciDomainID;
	}
	return _ERROR_;
};

/** Getter of major version release
 * @return major / _ERROR_
 */
int Cuda_GPU::get_major(void){
	if(this->assigned_gpu_id != _ERROR_){
		return this->fd_gpu.major;
	}
	return _ERROR_;
};

/** Getter of minor version release
 * @return minor / _ERROR_
 */
int Cuda_GPU::get_minor(void){
	if(this->assigned_gpu_id != _ERROR_){
		return this->fd_gpu.minor;
	}
	return _ERROR_;
};

/** Getter of Assigned GPU ID
 * @return Assigned GPU ID / _ERROR_
 */
int Cuda_GPU::get_gpu_id(void){
	if(this->assigned_gpu_id != _ERROR_){
		return this->assigned_gpu_id;
	}
	return _ERROR_;
};

