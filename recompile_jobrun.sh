#!/bin/bash

# This script "make clean", then "make" the program in ./source_code 
# and conditionaly "qsub" a job to PBSpro, with settings defined in pbs_script.scr .

# "make clean"
/usr/bin/make clean -C ./source_code/

# "make", continue only after success
/usr/bin/make -C ./source_code/
if [ $? -eq 0 ]; then
	echo "//==========================================================//"
	echo "//                      COMPILATION SUCCESS                 //"
	echo "//==========================================================//"
else
	echo "//*!*!*!*!*!*!*!*!*!*!*!*!*!*!*!*!*!*!*!*!*!*!*!*!*!*!*!*!*!//"
	echo "//                       COMPILATION ERROR                  //"
	echo "//*!*!*!*!*!*!*!*!*!*!*!*!*!*!*!*!*!*!*!*!*!*!*!*!*!*!*!*!*!//"
	exit 0
fi

# schedule the PBSpro job
qsub pbs_script.scr

# list the current PBSpro jobs
qstat -n
