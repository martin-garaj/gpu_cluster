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




#ifndef GPU_IN_GRID_CUH_
#define GPU_IN_GRID_CUH_



//==========================================================//
//						INCLUDES							//
//==========================================================//
// automatic Unified Memory allocation
#include "Managed.cuh"

// GPU configuration
#include "config_GPU.h"

// GPU configuration
#include "NN_def.hpp"

// Cereal - serialization
#include <cereal/archives/binary.hpp>
#include <cereal/archives/portable_binary.hpp>
//#include <string>
//#include <sstream>

//==========================================================//
//						STRUCTURES							//
//==========================================================//
/** every GPU element (block/thread) is provided with one of these structures
 */
class elements{
	//==========================//
	//			CONTENT			//
	//==========================//
	public:
		int grid_id;
		int NN_id;
		NN_def NN;
		// debugging parameter
		int debug;

	//==========================//
	//		SERIALIZATION		//
	//==========================//
		// function required by cereal library
		template<class Archive>
		void serialize(Archive & ar){
			ar( grid_id );
			ar( NN_id );
			ar( NN );
			ar( debug );
		}
};

//==========================================================//
//						OBJECT DEFINITION					//
//==========================================================//
/** Encapsulates all input data distributed to grid (elements = blocks/threads)
 */
class GPU_grid_in : public Managed{
	//==========================//
	//			CONTENT			//
	//==========================//
	public:
		int gpu_id;
		elements element[GRID_SIZE];

		// debugging parameter
		int debug;


	//==========================//
	//		SERIALIZATION		//
	//==========================//
		// function required by cereal library
		template<class Archive>
		void serialize(Archive & ar){
			ar( gpu_id );
			ar( debug );
			ar( element );
		}
};


#endif /* GPU_IN_GRID_CUH_ */
