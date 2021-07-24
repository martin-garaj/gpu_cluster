#!/bin/bash

# This script "qsub" a job to PBSpro, with settings defined in pbs_script.scr .

# schedule the PBSpro job
qsub pbs_script.scr

# list the current PBSpro jobs
qstat -n
