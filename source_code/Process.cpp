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


// class header
#include "Process.hpp"


/** Constructor
 */
Process::Process(int commRank){
	// remember the commRank
	process_id = commRank;

	// check for number of available CUDA devices
	this->total_gpu_available = this->gpu.dev_count();

	// preset these values to _ERROR_ to prevent misunderstandings
	//this->gpu.get_gpu_id() = _ERROR_;
	this->approx_node_id = _ERROR_;
}

/** Destructor
 */
Process::~Process(){

};



/** Returns the Process' identification number
 */
int Process::get_id(void){
	return process_id;
};

/** Rerun the number of available CUDA devices
 * @return number of CUDA devices
 */
int Process::availabe_gpu(void){
	return gpu.dev_count();
};

/** Automatically locates a GPU within a single computational cluster node, under 4 assumptions (check function body)
 * @return _SUCCESS_ / _ERROR_
 */
int Process::assign_gpu(void){
	// find* the GPU
	// * this process relies on 4 facts :
	//	1) the computational nodes have the same topology (every computational node
	//			has the same number of GPUs as any other computational node)
	//	2) the MPI processes are distributed by PBSpro in round-robin fashion
	//	3) CUDA labels the GPUs starting from 0, up to (# of available GPUs)-1, in other words, consecutively
	//	4) all the visible CUDA devices are available
	//
	// 	Because of the above assumptions, the (MPI)process_id can be used to assign unique GPU,
	//	so that no other MPI process will claim the same GPU twice (check the code below).

	int local_process_id; // MPI process id, local to the computational node
	int node_id;	// this is complementary information to local_process_id

	// calculate local process id
	local_process_id = (this->process_id-1) % (this->total_gpu_available);

	// keep the information which is lost when modulo-dividing
	node_id = (int ) ((this->process_id-1) / (this->total_gpu_available));

	// set the GPU
	if(this->gpu.assign(local_process_id) == _SUCCESS_){
		//this->gpu.assigned_gpu_id = local_process_id;
		this->approx_node_id = node_id;
		return _SUCCESS_;
	}

	std::cout << "MPI[" << this->process_id << "] _ERROR_ assign_gpu(void) : (" << local_process_id << "), # of available gpu : " << gpu.dev_count() << std::endl;
	return _ERROR_;

};


/** Assigns specified GPU to Process
 * @param gpu ID as provided by by CUDA (starts from 0, up to amount returned by dev_count()-1)
 */
int Process::assign_gpu(int gpu_id){
	return gpu.assign(gpu_id);
};

/** Getter for assigned GPU ID
 * @return returns the GPU ID as provide
 */
int Process::get_assigned_gpu_id(void){
	return this->gpu.get_gpu_id();
}

/** Getter for approximate node ID
 * @return returns the approximate ID of a computational node, for more information please check a Process::assign_gpu() body
 */
int Process::get_approx_node_id(void){
	return this->approx_node_id;
}


int Process::print_gpu(void){

	struct_fd_gpu temp_fd_gpu;

	if( this->gpu.get_fd_gpu(&temp_fd_gpu) == _SUCCESS_){
		std::string s_name(temp_fd_gpu.name);
		std::cout << "\tnumber of available GPU : " << this->total_gpu_available << std::endl;
		std::cout << "\tassigned GPU name       : " << s_name << std::endl;
		std::cout << "\tassigned GPU            : " << this->gpu.get_gpu_id() << std::endl;
		std::cout << "\tapproximated node #     : " << this->approx_node_id << std::endl;

		std::cout << "\t\tpciBusID               : " << temp_fd_gpu.pciBusID << std::endl;
		std::cout << "\t\tpciDeviceID           : " << temp_fd_gpu.pciDeviceID << std::endl;
		std::cout << "\t\tpciDomainID           : " << temp_fd_gpu.pciDomainID << std::endl;

		// print other important information from file descriptor of the GPU

		return _SUCCESS_;
	}

	std::cout << "MPI[" << this->process_id << "] _ERROR_ print_gpu(void) : CANNOT OBTAINE fd_gpu"  << std::endl;
	return _ERROR_;

};

int Process::run_kernel(int blocks, int threads, GPU_grid_in &mem_grid_in, GPU_grid_out &mem_grid_out, GPU_shared_in &mem_shared_in){
	// kernel
	//this->kernel.kernel_execute(blocks, threads, mem_grid_in, mem_grid_out, mem_shared_in);
	kernel_execute(blocks, threads, mem_grid_in, mem_grid_out, mem_shared_in);

	return _SUCCESS_;
};

int Process::cuda_synchronize(void){
	//this->kernel.kernel_synchronize();
	kernel_synchronize();
	return _SUCCESS_;
};


