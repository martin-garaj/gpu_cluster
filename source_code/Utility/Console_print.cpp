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
#include "Console_print.hpp"



int Console_print::limit_time(unsigned long long time_raw, unsigned int limit, float &time_limited, std::string &unit){

	if			( ( (time_raw/_D_) )	> limit){
		unit = "Day";
		time_limited = ((float)time_raw / _D_);
	}else if	( ( (time_raw/_h_) )	> limit){
		unit = "h";
		time_limited = ((float)time_raw / _h_);
	}else if	( ( (time_raw/_m_) )	> limit){
		unit = "m";
		time_limited = ((float)time_raw / _m_);
	}else if	( ( (time_raw/_s_) )	> limit){
		unit = "s";
		time_limited = ((float)time_raw / _s_);
	}else{
		unit = "ms";
		time_limited = ((float)time_raw);
	}

	return _SUCCESS_;
}

int Console_print::limit_time(float time_raw, unsigned int limit, float &time_limited, std::string &unit){

	if			( ( (time_raw/(float)_D_) )	> limit){
		unit = "Day";
		time_limited = (time_raw / (float)_D_);
	}else if	( ( (time_raw/(float)_h_) )	> limit){
		unit = "h";
		time_limited = (time_raw / (float)_h_);
	}else if	( ( (time_raw/(float)_m_) )	> limit){
		unit = "m";
		time_limited = (time_raw / (float)_m_);
	}else if	( ( (time_raw/(float)_s_) )	> limit){
		unit = "s";
		time_limited = (time_raw / (float)_s_);
	}else{
		unit = "ms";
		time_limited = (time_raw);
	}

	return _SUCCESS_;
}


int Console_print::limit_data(unsigned long long data_raw, unsigned int limit, float &date_limited, std::string &unit){

	if			( ( (data_raw/_TB_) )	> limit){
		unit = "TB";
		date_limited = ((float)data_raw / _TB_);
	}else if	( ( (data_raw/_GB_) )	> limit){
		unit = "GB";
		date_limited = ((float)data_raw / _GB_);
	}else if	( ( (data_raw/_MB_) )	> limit){
		unit = "MB";
		date_limited = ((float)data_raw / _MB_);
	}else if	( ( (data_raw/_KB_) )	> limit){
		unit = "KB";
		date_limited = ((float)data_raw / _KB_);
	}else{
		unit = "B";
		date_limited = ((float)data_raw);
	}

	return _SUCCESS_;
}

int Console_print::limit_data(float data_raw, unsigned int limit, float &date_limited, std::string &unit){

	if			( ( (data_raw/(float)_TB_) )	> limit){
		unit = "TB";
		date_limited = (data_raw / (float)_TB_);
	}else if	( ( (data_raw/(float)_GB_) )	> limit){
		unit = "GB";
		date_limited = (data_raw / (float)_GB_);
	}else if	( ( (data_raw/(float)_MB_) )	> limit){
		unit = "MB";
		date_limited = (data_raw / (float)_MB_);
	}else if	( ( (data_raw/(float)_KB_) )	> limit){
		unit = "KB";
		date_limited = (data_raw / (float)_KB_);
	}else{
		unit = "B";
		date_limited = (data_raw);
	}

	return _SUCCESS_;
}
