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



#include <cuda_runtime_api.h>
#include <cuda.h>

#ifndef MANAGED_INHERIT_CUH_
#define MANAGED_INHERIT_CUH_


//==========================================================//
//						CLASS DEFINITION					//
//==========================================================//

/** Inheriting from this object gives the object ability to get automatically
 * allocated in Unified Memory, when using "new" operator
 */
class Managed{
public:
	void *operator new(size_t len){
		void *ptr;
		cudaMallocManaged(&ptr, len);
		return ptr;
	}

	void operator delete(void *ptr){
		cudaFree(ptr);
	}
};


#endif /* MANAGED_INHERIT_CUH_ */