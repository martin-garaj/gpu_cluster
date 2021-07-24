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




#ifndef GPU_OUT_GRID_CUH_
#define GPU_OUT_GRID_CUH_


#include <cuda_runtime_api.h>
#include <cuda.h>

//==========================================================//
//						INCLUDES							//
//==========================================================//
// automatic Unified Memory allocation
#include "Managed.cuh"

// GPU configuration
#include "config_GPU.h"


//==========================================================//
//						OBJECT DEFINITION					//
//==========================================================//
/** encapsulates all input data distributed to grid (elements = blocks/threads)
 */
class GPU_grid_out : public Managed{
	//==========================//
	//			CONTENT			//
	//==========================//
	public:
		int gpu_id;
		float result[GRID_SIZE];
		// debugging parameter
		int debug;

	//==========================//
	//		SERIALIZATION		//
	//==========================//
		// function required by cereal library
		template<class Archive>
		void serialize(Archive & ar){
			ar( gpu_id );
			ar( cereal::binary_data( result , sizeof(float) * GRID_SIZE));
			ar( debug );
		}
};



#endif /* GPU_OUT_GRID_CUH_ */
