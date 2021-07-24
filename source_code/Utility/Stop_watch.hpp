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


#ifndef STOP_WATCH_HPP_
#define STOP_WATCH_HPP_


// time measurement
#include <chrono>

// vector type
#include <vector>


class Stop_watch{
public:
	Stop_watch();
	~Stop_watch();
	int start(void);
	int stop(void);
	int stop(bool record);
	float elapsed_ms(void);
	int mem_count(void);
	int clear_mem(void);
	float average(void);
	float average(float &deviation, float &total_time);
private:
	bool running;
	float elapsed_time_ms;
	std::chrono::time_point<std::chrono::system_clock> time_start;
	std::chrono::time_point<std::chrono::system_clock> time_stop;
	std::vector<float> memory;
};


#endif /* STOP_WATCH_HPP_ */
