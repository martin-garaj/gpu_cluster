# gpu_cluster
Boiler-plate framework for job scheduling on HPC GPU cluster.

This was a fun to program, as I have gained some experience with:
  - _template programming_ in C++
  - _CUDA programming_ on NVIDIA GPUs
  - _object serialization with Cereal library_ to prepare data for transmission
  - _OpenMPI transmission and synchronization_ for sending/receiving data
  - _PBS Pro scripting_ for job distribution on cluster
  - general _Object Oriented Programming_ concepts to not get lost on the way :D

# File description
[pbs_script.scr](./gpu_cluster/pbs_script.scr): PBS Pro script to allocate resource and run the user program in ./source_code/test_framework ([line 109](https://github.com/martin-garaj/gpu_cluster/blob/fcde60d0c0ebed684a9ed1386eee799844226eda/pbs_script.scr#L109))

