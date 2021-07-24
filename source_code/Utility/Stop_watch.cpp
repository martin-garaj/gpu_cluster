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
#include "Stop_watch.hpp"


Stop_watch::Stop_watch(){
	this->running  = false;
	this->elapsed_time_ms = 0.0;
};

Stop_watch::~Stop_watch(){};

int Stop_watch::start(void){
	if(this->running == true){
		return 0;
	}else{
		this->time_start = std::chrono::system_clock::now();
		this->running = true;
	}
	return 1;
};

int Stop_watch::stop(void){
	if(this->running == false){
		return 0;
	}else{
		this->time_stop = std::chrono::system_clock::now();
		this->running = false;
		this->elapsed_time_ms = (float) std::chrono::duration_cast<std::chrono::milliseconds>(this->time_stop - this->time_start).count();
	}
	return 1;
};

int Stop_watch::stop(bool record){
	if(this->running == false){
		return 0;
	}else{
		this->time_stop = std::chrono::system_clock::now();
		this->running = false;
		this->elapsed_time_ms = (float) std::chrono::duration_cast<std::chrono::milliseconds>(this->time_stop - this->time_start).count();

		if(record){
			this->memory.push_back(this->elapsed_time_ms);
		}
	}
	return 1;
};

float Stop_watch::elapsed_ms(void){
	return this->elapsed_time_ms;
};

int Stop_watch::mem_count(void){
	return this->memory.size();
};

int Stop_watch::clear_mem(void){
	this->memory.clear();

};

float Stop_watch::average(void){
	float average = 0.0;
	for(int i = 0; i<this->memory.size(); i++){
		average = average + ( this->memory.at(i)/this->memory.size() );
	}

	return average;
};


float Stop_watch::average(float &deviation, float &total_time){
	float average = 0.0;
	float temp_total_time;
	float temp_time;
	int num_of_measurements;

	num_of_measurements = this->memory.size();
	for(int i = 0; i<this->memory.size(); i++){
		temp_time = this->memory.at(i);
		average = average + ( temp_time/num_of_measurements );
		temp_total_time = temp_total_time + temp_time;
	}
	total_time = temp_total_time;

	float temp_deviation = 0.0;
	for(int i = 0; i<this->memory.size(); i++){
		temp_time = this->memory.at(i);
		temp_deviation = (temp_time - average) * (temp_time - average);
	}
	deviation = temp_deviation;


	return average;

};

