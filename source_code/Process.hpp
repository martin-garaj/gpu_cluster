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

#ifndef PROCESS_HPP_
#define PROCESS_HPP_

//==========================================================//
//						INCLUDES							//
//==========================================================//
// return values
#include "return_values.h"

// strncpy
#include <cstring>

// Cuda GPU object
#include "Cuda_GPU.cuh"

// Cuda kernel functions
#include "Cuda_kernel.cuh"

// c++ serialization
#include <iostream>
#include <string>
#include <sstream>

// cereal serialization
#include <cereal/archives/binary.hpp>
#include <cereal/archives/portable_binary.hpp>

// std::cout
#include <iostream>

// MPI
#include <mpi.h>


//==========================================================//
//						MACRO DEFINITION					//
//==========================================================//
// Error handling macros
#define MPI_CHECK(call) \
    if((call) != MPI_SUCCESS) { \
        std::cerr << "MPI error calling \""#call"\"\n"; \
        MPI_Abort(MPI_COMM_WORLD, -1); }


//==========================================================//
//						OBJECT DEFINITION					//
//==========================================================//
class Process{
	//==========================//
	//			FUNCTIONS		//
	//==========================//
	public:
		Process(int commRank);
		~Process();
		int get_id(void);
		int availabe_gpu(void);
		int assign_gpu(void);
		int assign_gpu(int gpu_id);
		int print_gpu(void);
		int get_assigned_gpu_id(void);
		int get_approx_node_id(void);
		int run_kernel(int blocks, int threads, GPU_grid_in &mem_grid_in, GPU_grid_out &mem_grid_out, GPU_shared_in &mem_shared_in);
		int cuda_synchronize(void);

		template<typename data_type>
		int serialize(data_type * data, char *serial_data, int *size);
		template<typename data_type>
		int deserialize(data_type * data, char *serial_data, int size);
		template<typename data_type>
		int do_both(data_type * data_in, data_type * data_out);

		// MPI functions
		template<typename data_type>
		int get_size(data_type * data);
		template<typename data_type>
		int send_to(data_type * data, int destination, int tag);
		template<typename data_type>
		int receive_from(data_type * data, int source, int tag);


	//==========================//
	//			CONTENT			//
	//==========================//
	private:
		Cuda_GPU gpu;
		int process_id;
		//int assigned_gpu_id;
		int approx_node_id;
		int total_gpu_available;

		//char mpi_name[256];
};


//======================================================================//
//					<TEMPLATE> FUNCTION DEFINITIONS						//
//======================================================================//

/** Calculates the necessary length of a char array, that is supposed to store the binary stream of provided data_type
 * @param an object to be stored in an array after serialization
 * @return necessary length of a char array to store the binary serialized object
 */
template<typename data_type>
int Process::get_size(data_type * data){

	// serialize the data
	std::ostringstream oss(std::ios::binary);
	cereal::BinaryOutputArchive ar(oss);
	ar(*data);
	std::string s=oss.str();

	// length of a string = number of symbols in the string (without terminating character '\0')
	// length of char array = number of symbols in the array + terminating character '\0'
	return s.length()+1;
};


template<typename data_type>
int Process::send_to(data_type * data, int destination, int tag){

	// serialize the data
	std::ostringstream oss(std::ios::binary);
	cereal::BinaryOutputArchive ar(oss);
	ar(*data);
	std::string s=oss.str();

	// MPI send
	MPI_CHECK(
		MPI_Send(
			(void*)s.c_str(),
			s.size(),
			MPI_CHAR,
			destination,
			tag,
			MPI_COMM_WORLD
	    )
	);

	return s.size();
};

template<typename data_type>
int Process::receive_from(data_type * data, int source, int tag){
	// get the length of data in a message
	MPI_Status status;
	status.MPI_SOURCE=source;
	MPI_Probe(source, tag, MPI_COMM_WORLD, &status);
	int n;
	MPI_Get_count(&status, MPI_CHAR, &n);

	// create temporary buffer to store the received data
	char * buf=new char[n];

	// store the received data in the buffer
	int count = MPI_Recv(
		    buf,
		    n,
		    MPI_CHAR,
		    source,
		    tag,
		    MPI_COMM_WORLD,
		    &status);

	std::istringstream iss(std::string(buf,n), std::ios::binary);
	cereal::BinaryInputArchive arin(iss);
	arin(*data);

	// clean buffer
	delete[] buf;
	return n;
};


/** Serializes the object into a binary stream of characters, and stores it into a pointer to provided char array
 * @param pointer to an object to be serialized
 * @param pointer to a char array storing the binary stream of characters
 * @param pointer to an integer, storing the length of the binary stream
 * @return _SUCCESS_
 */
template<typename data_type>
int Process::serialize(data_type * data, char *serial_data, int *size){

	// serialize the data
	std::ostringstream oss(std::ios::binary);
	cereal::BinaryOutputArchive ar(oss);
	ar(*data);

	// store the binary stream in a string
	std::string string_data=oss.str();
	*size = string_data.length()+1;

	// copy the string into a provided pointer to a char array
	string_data.copy(serial_data, string_data.length(), 0);
	serial_data[string_data.length()] = '\0';

	// return
	return _SUCCESS_;
};

/** Provides the mechanism of transforming binary stream into an object
 * @param pointer to an object to store the deserialized data
 * @param pointer to a char array storing the binary stream of serialized object
 * @param integer of length of the char array storing the binary stream of serialized object
 * @return _SUCCESS_
 */
template<typename data_type>
int Process::deserialize(data_type * data, char *serial_data, int size){
	// transform the char array to a binary stream
	std::istringstream iss(std::string(serial_data, size), std::ios::binary);
	// cereal deserialization
	cereal::BinaryInputArchive arin(iss);
	arin(*data);

	return _SUCCESS_;
};

#endif /* PROCESS_HPP_ */
