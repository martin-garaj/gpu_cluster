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

#ifndef GPU_IN_SHARED_CUH_
#define GPU_IN_SHARED_CUH_

//==========================================================//
//						PARAMETERS							//
//==========================================================//
#define test_cases	2000	// number of input/output reference voltage pairs


//==========================================================//
//						INCLUDES							//
//==========================================================//
// automatic Unified Memory allocation
#include "Managed.cuh"

// GPU configuration
#include "config_GPU.h"

//==========================================================//
//						STRUCTURES							//
//==========================================================//
/** Encapsulates the parameters of a dynamic model,
 * the model is governed by difference equations and evaluated by 2nd order Runge-Kutta method.
 */
class struct_model{
	//==========================//
	//			CONTENT			//
	//==========================//
	public:
		float i_L;
		float c_V;
		// debugging parameter
		int debug;

	//==========================//
	//		SERIALIZATION		//
	//==========================//
		// function required by cereal library
		template<class Archive>
		void serialize(Archive & ar){
			ar( i_L );
			ar( c_V );
			ar( debug );
		}
};

/** Encapsulates all the information regarding different scenarios
 * for training/evaluating the performance of NN control.
 */
class struct_cases{
public:
	//==========================//
	//			CONTENT			//
	//==========================//
	float input_ref[test_cases];
	float output_ref[test_cases];
	// debugging parameter
	int debug;

	//==========================//
	//		SERIALIZATION		//
	//==========================//
	// function required by cereal library
	template<class Archive>
	void serialize(Archive & ar){
		ar( cereal::binary_data( input_ref , sizeof(float) * test_cases) );
		ar( cereal::binary_data( output_ref , sizeof(float) * test_cases) );
		ar( debug );
	}
};


//==========================================================//
//						OBJECT DEFINITION					//
//==========================================================//
/** Encapsulates all input data distributed to grid (elements = blocks/threads)
 */
class GPU_shared_in : public Managed{
public:
	//==========================//
	//			CONTENT			//
	//==========================//
	struct_model model;
	struct_cases test_case;
	// debugging parameter
	int debug;

	//==========================//
	//		SERIALIZATION		//
	//==========================//
	// function required by cereal library
	template<class Archive>
	void serialize(Archive & ar){
		ar( model );
		ar( test_case );
	}
};


#endif /* GPU_IN_SHARED_CUH_ */
