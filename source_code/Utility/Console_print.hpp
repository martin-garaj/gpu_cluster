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

#ifndef CONSOLE_PRINT_HPP_
#define CONSOLE_PRINT_HPP_

// std::string
#include <string>

// return values
#include "return_values.h"

class Console_print{
public:
	static int limit_time(unsigned long long time_raw, unsigned int limit, float &time_limited, std::string &unit);
	static int limit_data(unsigned long long data_raw, unsigned int limit, float &date_limited, std::string &unit);
	static int limit_time(float time_raw, unsigned int limit, float &time_limited, std::string &unit);
	static int limit_data(float data_raw, unsigned int limit, float &date_limited, std::string &unit);

	// time constants
	static const long int _ms_ = 1;
	static const long int _s_ 	= 1000;
	static const long int _m_ 	= 60ull*1000;
	static const long int _h_ 	= 60ull*60*1000;
	static const long int _D_ 	= 24ull*60*60*1000;

	// data constants
	static const long int _B_ = 1;
	static const long int _KB_ = 1024;
	static const long int _MB_ = 1024ull*1024;
	static const long int _GB_ = 1024ull*1024*1024;
	static const long int _TB_ = 1024ull*1024*1024*1024;

};


#endif /* CONSOLE_PRINT_HPP_ */
