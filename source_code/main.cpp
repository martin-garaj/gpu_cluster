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


//==========================================================//
//						INCLUDES							//
//==========================================================//
// configuration
#include "config.h"
// return values
#include "return_values.h"
// constants / ROOT_PROCESS
#include "const.h"
// process object
#include "Process.hpp"
// data objects
#include "Data_objects/GPU_grid_in.cuh"
#include "Data_objects/GPU_grid_out.cuh"
#include "Data_objects/GPU_shared_in.cuh"
// time measurement
#include "Stop_watch.hpp"
// Console printing
#include "Console_print.hpp"


// std::cout
#include <iostream>
// usleep()
#include <unistd.h>
// MPI
#include <mpi.h>
// NAN constant
#include <cmath>
// std::string
#include <string>
// current time
#include <ctime>
#include <ratio>


//==========================================================//
//						PARAMETERS							//
//==========================================================//
// this file is meant as supplement of run-time parameter passing mechanism,
// most likely implemented as XML parser with appropriate structure for passing the parameters

// define the MPI process ID to be tested (0 = ROOT PROCESS, therefore select any MPI process above 0)
const int TIME_TESTING_PROCESS[] = {1, 2, 3, 4, 5, 6, 7, 8};
const int TIME_TESTING_PROCESS_COUNT = sizeof(TIME_TESTING_PROCESS)/sizeof(TIME_TESTING_PROCESS[0]);
int PROCESS_ID;

int CORRECT_RESULT_0 = 96.0;
int CORRECT_RESULT_1 = 160.0;

    //==========================================================================================//
    //										GLOBAL VATIABLES										//
    //==========================================================================================//

	float total_time;
	float deviation;
	float average;
	bool even_turn;
	unsigned long long total_data_sent;
	unsigned long long total_data_received;
	unsigned long long total_data_transfered;
	unsigned long long total_error;
	unsigned long long total_message;
	unsigned long long size_of_grid_in;
	unsigned long long size_of_grid_out;
	unsigned long long size_of_shared_in;
	std::chrono::system_clock::time_point starting_time;
	std::chrono::system_clock::time_point ending_time;
	std::chrono::duration<int,std::ratio<60*60*24> > one_day (1);
	time_t tt;
	float temp_time;
	float temp_data;
	std::string temp_unit;

    //==========================================================================================//
    //										CONSTANTS										//
    //==========================================================================================//
    		static const float weights_set_0[NN_layers][NN_neurons][NN_weights] = 	{ 	// NN
    													{ 	// input layer
    														{0.0,1.0,2.0,3.0,4.0},
    														{0.0,1.0,2.0,3.0,4.0},
    														{0.0,1.0,2.0,3.0,4.0},
    														{0.0,1.0,2.0,3.0,4.0}
    													},
    													{ 	// hidden layer 0
															{0.0,1.0,2.0,3.0,4.0},
															{0.0,1.0,2.0,3.0,4.0},
															{0.0,1.0,2.0,3.0,4.0},
															{0.0,1.0,2.0,3.0,4.0}
    													},
    													{ 	// hidden layer 1
															{0.0,1.0,2.0,3.0,4.0},
															{0.0,1.0,2.0,3.0,4.0},
															{0.0,1.0,2.0,3.0,4.0},
															{0.0,1.0,2.0,3.0,4.0}
    													},
    													{ 	// output layer
															{0.0,1.0,2.0,3.0,4.0},
															{0.0,1.0,2.0,3.0,4.0},
															{0.0,1.0,2.0,3.0,4.0},
															{0.0,1.0,2.0,3.0,4.0}
    													}
    											};

		static const float weights_set_1[NN_layers][NN_neurons][NN_weights] = 	{ 	// NN
    													{ 	// input layer
    														{4.0,3.0,2.0,1.0,0.0},
    														{4.0,3.0,2.0,1.0,0.0},
    														{4.0,3.0,2.0,1.0,0.0},
    														{4.0,3.0,2.0,1.0,0.0}
    													},
    													{ 	// hidden layer 0
															{4.0,3.0,2.0,1.0,0.0},
															{4.0,3.0,2.0,1.0,0.0},
															{4.0,3.0,2.0,1.0,0.0},
															{4.0,3.0,2.0,1.0,0.0}
    													},
    													{ 	// hidden layer 1
															{4.0,3.0,2.0,1.0,0.0},
															{4.0,3.0,2.0,1.0,0.0},
															{4.0,3.0,2.0,1.0,0.0},
															{4.0,3.0,2.0,1.0,0.0}
    													},
    													{ 	// output layer
															{4.0,3.0,2.0,1.0,0.0},
															{4.0,3.0,2.0,1.0,0.0},
															{4.0,3.0,2.0,1.0,0.0},
															{4.0,3.0,2.0,1.0,0.0}
    													}
    											};
											


    //==========================================================================================//
    //										MAIN											//
    //==========================================================================================//
int main(int argc, char *argv[]){
	//--------------------------------INITIALIZE_MPI--------------------------------//
    MPI_CHECK(MPI_Init(&argc, &argv));


    //--------------------------------VARIABLES--------------------------------//
	// MPI variables
	int commSize;
	int commRank;

	//--------------------------------EXECUTION--------------------------------//
	// Get MPI node number and node count
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &commSize));
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &commRank));

    //--------------------------------SANITY CHECK--------------------------------//
    for(int xyz = 0; xyz < TIME_TESTING_PROCESS_COUNT; xyz++){
    	PROCESS_ID = TIME_TESTING_PROCESS[xyz];
    	if(PROCESS_ID >=  commSize){
    		std::cout << " NOT ENOUGH PROCESSES TO RUN THE TEST" << std::endl;
    		std::cout << " THE HIGHEST PROCESS RANK IS (commSize-1) = "<< commSize-1 << " THERE IS PROCESS WITH RANK = " << PROCESS_ID << std::endl;
    		MPI_CHECK( MPI_Finalize() );
    		return 0;
    	}
    }

    // this is the same among all MPI processes
    Process Process(commRank);
    Stop_watch timer_0;
    Stop_watch timer_1;
    Stop_watch timer_2;
	Stop_watch timer_3;
	Stop_watch timer_4;
 
    //==========================================================================================//
    //										ROOT PROCESS										//
    //==========================================================================================//
    if(commRank == ROOT_PROCESS){
	starting_time = std::chrono::system_clock::now();
		// timer_4 - measures whole execution on the ROOT process
		timer_4.start();

		// Objects passed by MPI
    	GPU_grid_in cpu_grid_in[2];
    	GPU_grid_out cpu_grid_out[TIME_TESTING_PROCESS_COUNT];
    	GPU_shared_in cpu_shared_in[TIME_TESTING_PROCESS_COUNT];

    	//--------------------------------STATISTICAL_VARIABLES--------------------------------//
        total_time = 0.0;
        deviation = 0.0;
        average = 0.0;
    	even_turn = true;
    	total_data_sent = 0;
    	total_data_received = 0;
    	total_data_transfered = 0;
    	total_error = 0;
    	total_message = 0;
    	size_of_grid_in = 0;
    	size_of_grid_out = 0;
    	size_of_shared_in = 0;


    	//--------------------------------DATA_INITIALIZATION--------------------------------//
    	// timer_0 - data initialization
    	timer_0.start();

    	size_of_grid_in = Process.get_size(&cpu_grid_in[0]);
    	size_of_grid_out = Process.get_size(&cpu_grid_out[0]);
    	size_of_shared_in = Process.get_size(&cpu_shared_in[0]);

    	// GPU_grid_in[0]
    	cpu_grid_in[0].debug = 9;
    	cpu_grid_in[0].gpu_id = -1;
    	// fill in every grid element
    	for(int i = 0; i<GRID_SIZE; i++){
    		cpu_grid_in[0].element[i].debug = 99;
    		cpu_grid_in[0].element[i].NN_id = i;
    		cpu_grid_in[0].element[i].grid_id = i;
    		cpu_grid_in[0].element[i].NN.debug = 999;
    		memcpy(cpu_grid_in[0].element[i].NN.weight, weights_set_0, sizeof(cpu_grid_in[0].element[i].NN.weight));
    	}


    	// GPU_grid_in[1]
    	cpu_grid_in[1].debug = 9;
    	cpu_grid_in[1].gpu_id = -1;
    	// fill in every grid element
    	for(int i = 0; i<GRID_SIZE; i++){
    		cpu_grid_in[1].element[i].debug = 99;
    		cpu_grid_in[1].element[i].NN_id = i;
    		cpu_grid_in[1].element[i].grid_id = i;
    		cpu_grid_in[1].element[i].NN.debug = 999;
    		memcpy(cpu_grid_in[1].element[i].NN.weight, weights_set_1, sizeof(cpu_grid_in[1].element[i].NN.weight));
    	}

    	// Initialize data
        for(int xyz = 0; xyz < TIME_TESTING_PROCESS_COUNT; xyz++){
        	// GPU_grid_out
        	cpu_grid_out[xyz].debug = 9999;
        	cpu_grid_out[xyz].gpu_id = -1;
        	for(int i = 0; i<GRID_SIZE; i++){
        		cpu_grid_out[xyz].result[i] = NAN;
        	}

        	// GPU_shared_in
        	cpu_shared_in[xyz].debug = 99999;
        	// others are not important
        }

    	// measure time for data initialization
    	timer_0.stop();
        //std::cout << "MPI[" << Process.get_id() << "]     data initialization : " << timer_0.elapsed_ms() << " [ms]"<< std::endl;

        //--------------------------------MPI_SYNCRONIZATION_0--------------------------------//
    	MPI_CHECK( MPI_Barrier(MPI_COMM_WORLD) );

    	// Receive status message
    	//	NOT IMPLEMENTED YET


        //--------------------------------MPI_SYNCRONIZATION_1--------------------------------//
		MPI_CHECK( MPI_Barrier(MPI_COMM_WORLD) );

		// Send the shared_in data to MPI slave processes
		for(int xyz = 0; xyz < TIME_TESTING_PROCESS_COUNT; xyz++){
			PROCESS_ID = TIME_TESTING_PROCESS[xyz];

			total_data_sent += (unsigned long long) Process.send_to(&cpu_shared_in[xyz], PROCESS_ID, TAG_SHARED_IN);
			total_message++;
		}// send data loop

		// Send/Gather the grid_in data in round-robin fashion
		for(int i = 0; i < TIME_TESTING_ITERATIONS; i++){
			// start time measurement
			timer_2.start();

			for(int xyz = 0; xyz < TIME_TESTING_PROCESS_COUNT; xyz++){
				PROCESS_ID = TIME_TESTING_PROCESS[xyz];

				// Keep altering the data
				if(even_turn){
					// send the data to MPI[PROCESS_ID]
					total_data_sent += (unsigned long long) Process.send_to(&cpu_grid_in[0], PROCESS_ID, TAG_GRID_IN);
					total_message++;
				}else{ // odd turn
					// send the data to MPI[PROCESS_ID]
					total_data_sent += (unsigned long long) Process.send_to(&cpu_grid_in[1], PROCESS_ID, TAG_GRID_IN);
					total_message++;
				}// send data loop

			}

			//--------------------------------MPI_SYNCRONIZATION_2--------------------------------//
			MPI_CHECK( MPI_Barrier(MPI_COMM_WORLD) );


			//  Receive & Check the grid_out data in round-robin fashion
			for(int xyz = 0; xyz < TIME_TESTING_PROCESS_COUNT; xyz++){
				PROCESS_ID = TIME_TESTING_PROCESS[xyz];

				// Receive the grid_out from MPI[PROCESS_ID]
				total_data_received += (unsigned long long) Process.receive_from(&cpu_grid_out[xyz], PROCESS_ID, TAG_GRID_OUT);
				total_message++;

				/*
				if(even_turn){
					std::cout << "MPI[" << Process.get_id() << "] results from MPI[" << PROCESS_ID <<  "] = " << cpu_grid_out[xyz].result[5] << std::endl;
					std::cout << "even_turn, even result = " << CORRECT_RESULT_0 << std::endl;
				}else{
					std::cout << "MPI[" << Process.get_id() << "] results from MPI[" << PROCESS_ID <<  "] = " << cpu_grid_out[xyz].result[5] << std::endl;
					std::cout << "even_turn, even result = " << CORRECT_RESULT_1 << std::endl;
				}
*/


				// Check the results
				if(even_turn){
					for(int k = 0 ; k < GRID_SIZE ; k++){
						if(cpu_grid_out[xyz].result[k] == CORRECT_RESULT_0){
							// EVERYTHING IS OK
						}else{
							total_error++;
						}
					}
				}else{
					for(int k = 0 ; k < GRID_SIZE ; k++){
						if(cpu_grid_out[xyz].result[k] == CORRECT_RESULT_1){
							// EVERYTHING IS OK
						}else{
							total_error++;
						}
					}
				}

			}// receive & check data loop

			even_turn = !even_turn;

			// stop time measurement - remember the measured time (true)
			timer_2.stop(true);

		}// time testing loop

		// timer_4 - the end of ROOT execution
		timer_4.stop();
	//==========================================================================================//
	//										SLAVE PROCESS										//
	//==========================================================================================//
    }else{

        //--------------------------------MPI_SYNCRONIZATION_0--------------------------------//
    	MPI_CHECK( MPI_Barrier(MPI_COMM_WORLD) );

		// Acknowledge MPI process existance
    	usleep(Process.get_id() * 100000);
    	std::cout << "MPI[" << Process.get_id() << "]     I am running ..." << std::endl;

    	// Assign GPU
        if( Process.assign_gpu() != _SUCCESS_){
        	std::cout << "MPI[" << Process.get_id() << "]     assign_gpu() _ERROR_" << std::endl;
        	MPI_Abort(MPI_COMM_WORLD, 777);
        	return _ERROR_;
        }else{
            std::cout << "MPI[" << Process.get_id() << "]     assigned GPU ID : "<< Process.get_assigned_gpu_id() <<", approx NODE ID : " << Process.get_approx_node_id() << std::endl;
        }// assign GPU end

    	// GPU HAS TO BE ASSIGNED PRIOR TO ALLOCATING GPU MEMORY, AS THE MEMORY IS ALLOCATED IN SHARED-CPU-GPU-MEMORY
    	GPU_grid_in *gpu_grid_in = new GPU_grid_in;
    	GPU_grid_out *gpu_grid_out = new GPU_grid_out;
    	GPU_shared_in *gpu_shared_in = new GPU_shared_in;


     	// Send status message
        //	NOT IMPLEMENTED YET



        //--------------------------------MPI_SYNCRONIZATION_1--------------------------------//
    	MPI_CHECK( MPI_Barrier(MPI_COMM_WORLD) );


    	timer_1.start();

		// Receive GPU_SHARED_MEM
		Process.receive_from(gpu_shared_in, ROOT_PROCESS, TAG_SHARED_IN);
		// Spread/Gather the data in round-robin fashion
		for(int i = 0; i < TIME_TESTING_ITERATIONS; i++){
			// Receive the data to be processed
			Process.receive_from(gpu_grid_in, ROOT_PROCESS, TAG_GRID_IN);
			timer_1.start();
			// run the kernel on GPU
			Process.run_kernel(BLOCKS, THREADS, *gpu_grid_in, *gpu_grid_out, *gpu_shared_in);
			Process.cuda_synchronize();
			timer_1.stop(true);

			//--------------------------------MPI_SYNCRONIZATION_2--------------------------------//
			MPI_CHECK( MPI_Barrier(MPI_COMM_WORLD) );
			// send the result
			Process.send_to(gpu_grid_out, ROOT_PROCESS, TAG_GRID_OUT);
		}// time testing loop

		timer_1.stop();




    }// MPI SLAVE PROCESS




    std::cout << "MPI[" << Process.get_id() << "]     I have finished ..." << std::endl;

	//==========================================================================================//
	//										MPI END												//
	//==========================================================================================//
    MPI_CHECK( MPI_Finalize() );






	//==========================================================================================//
	//										STATISTICS											//
	//==========================================================================================//
	if(commRank == ROOT_PROCESS){
		ending_time = std::chrono::system_clock::now();

		tt = std::chrono::system_clock::to_time_t ( starting_time );
  		std::cout << "PROGRAM STARTING TIME : " << ctime(&tt);

		//--------------------------------MPI_&_CUDA--------------------------------//
		std::cout << "MPI[" << Process.get_id() << "]     MPI & CUDA Statictics :" << std::endl;
		std::cout << "               num of tested MPI Proc : " << TIME_TESTING_PROCESS_COUNT << " " << std::endl;
		std::cout << "               CUDA grid size         : " << GRID_SIZE << " <<<" << BLOCKS << "*" << THREADS << ">>>" << std::endl;
		std::cout << "               total num of kernels   : " << TIME_TESTING_PROCESS_COUNT*GRID_SIZE << std::endl;

		//--------------------------------DATA--------------------------------//
		std::cout << "MPI[" << Process.get_id() << "]     Data Statictics :" << std::endl;
		// SIZE - GRID IN
		Console_print::limit_data(size_of_grid_in, 10, temp_data, temp_unit);
		std::cout << "               grid_in size       : " << temp_data << " ["<< temp_unit <<"]" << std::endl;
		// SIZE - GRID OUT
		Console_print::limit_data(size_of_grid_out, 10, temp_data, temp_unit);
		std::cout << "               grid_out size      : " << temp_data << " ["<< temp_unit <<"]" << std::endl;
		// SIZE - SHARED IN
		Console_print::limit_data(size_of_shared_in, 10, temp_data, temp_unit);
		std::cout << "               shared_in size     : " << temp_data << " ["<< temp_unit <<"]" << std::endl;
		// SIZE - DATA TRANSFERED
		total_data_transfered = total_data_sent + total_data_received;
		Console_print::limit_data(total_data_transfered, 10, temp_data, temp_unit);
		std::cout << "               data transfered    : " << temp_data << " ["<< temp_unit <<"]" << std::endl;
		// # ERRORS
		std::cout << "               errorous messages  : " << total_error / GRID_SIZE << " instances" << std::endl;
		// # ERROR MESSAGES
		std::cout << "               number of messages : " << total_message << " instances" << std::endl;

		//--------------------------------TIMING--------------------------------//
		std::cout << "MPI[" << Process.get_id() << "]     Timing Statictics :" << std::endl;


		// TIME - DATA INITIALIZATION
		Console_print::limit_time(timer_0.elapsed_ms(), 10, temp_data, temp_unit);
		std::cout << "               init data            : " << temp_data << " ["<< temp_unit <<"]" << std::endl;

		
		// TIME - AVERAGE EXECUTION
		average = timer_2.average(deviation, total_time);
		Console_print::limit_time(average, 10, temp_data, temp_unit);
		std::cout << "               average execution    : " << temp_data << " ["<< temp_unit <<"]" << std::endl;
		// TIME - DEVIATION
		Console_print::limit_time(deviation, 10, temp_data, temp_unit);
		std::cout << "               deviation execution  : " << temp_data << " ["<< temp_unit <<"]" << std::endl;

		// TIME - TOTAL EXECUTION
		Console_print::limit_time(total_time, 10, temp_data, temp_unit);
		std::cout << "               total execution      : " << temp_data << " ["<< temp_unit <<"]" << std::endl;


		// TIME - TORAL ROOT TIME
		Console_print::limit_time(timer_4.elapsed_ms(), 10, temp_data, temp_unit);
		std::cout << "               total ROOT time      : " << temp_data << " ["<< temp_unit <<"]" << std::endl;

		tt = std::chrono::system_clock::to_time_t ( ending_time );
  		std::cout << "PROGRAM ENDING TIME : " << ctime(&tt);
	}


	return 0;
}

