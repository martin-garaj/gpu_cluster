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


#ifndef NN_DEF_H_
#define NN_DEF_H_

//==========================================================//
//						INCLUDES							//
//==========================================================//
// std::cout
#include <iostream>

// Cereal - serialization
#include <cereal/archives/binary.hpp>
#include <cereal/archives/portable_binary.hpp>
#include <string>
#include <sstream>

//==========================//
//			CONSTANTS		//
//==========================//
#define NN_layout  			{4,4,4,4}	// E.G {2, 5, 2, 1} is NN with :
								//		inputs layer   : 2 inputs
								//		hidden layer 1 : 5 neurons
								// 		hidden layer 2 : 2 neurons
								// 		output layer   : 1 output
#define NN_layers 			4				// including input+hidden+output layers
#define NN_neurons 			4				// keep in mind that "neurons >= MAX(layout[])"
#define NN_weights 			((NN_neurons+1)*NN_neurons)	// including bias


//==========================================================//
//						OBJECT DEFINITION					//
//==========================================================//
class NN_def{


	//==========================//
	//			CONTENT			//
	//==========================//
	public:

		float weight[NN_layers][NN_neurons][NN_weights];
		// ensure the size of NN_layout equals the NN_layer parameter
		int layout[NN_layers] = NN_layout;

		// debugging parameter
		int debug;



	//==========================//
	//		SERIALIZATION		//
	//==========================//
	// function required by cereal library
	template<class Archive>
	void serialize(Archive & ar){
		ar( cereal::binary_data( weight , sizeof(float) * NN_layers * NN_neurons * NN_weights));
		ar( cereal::binary_data( layout , sizeof(int) * NN_layers) );
		ar( debug );
	}
};



#endif /* NN_DEF_H_ */
