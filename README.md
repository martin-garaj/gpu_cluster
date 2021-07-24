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
## PBS scripts
[pbs_script.scr](./gpu_cluster/pbs_script.scr): PBS Pro script to allocate resources within cluster and run the executable ([line 109](https://github.com/martin-garaj/gpu_cluster/blob/fcde60d0c0ebed684a9ed1386eee799844226eda/pbs_script.scr#L109))

## C++ (CPU)
[main.cpp](./gpu_cluster/source_code/main.cpp): Main file representing _main_ and _worker_ nodes
  - _main_ node is differentiated by [const int] **commRank == ROOT_PROCESS** (defined in [const.h](./gpu_cluster/source_code/const.h))
  - _worker_ nodes are all other nodes with **commRank != ROOT_PROCESS**
  
[Process.cpp](./gpu_cluster/source_code/Process.cpp), [Process.hpp](./gpu_cluster/source_code/Process.hpp): Object representing both _main_ and _worker_ nodes (NOTICE: the same kind of _Process_ object represents different kinds of nodes)
  - NOTICE: The object has a regular OOP structure:
    - [Process.cpp](./gpu_cluster/source_code/Process.cpp) contains implementations of classes and functions 
    - [Process.hpp](./gpu_cluster/source_code/Process.hpp) contains declarations + implementations of [template functions](https://github.com/martin-garaj/gpu_cluster/blob/45a0ebc99051b16a3dbca8e8fcef00032a10187a/source_code/Process.hpp#L114) (template functions cannot be directly compiled from implementation in .hpp file)


## [Unified memory](./gpu_cluster/source_code/Data_objects/) (CUDA<->C++)
[Managed.cuh](./gpu_cluster/source_code/Data_objects/Managed.cuh): Inheriting from this class allows the object to be _unified memory_ (memory on GPU that is visible from CPU)
  - [GPU_grid_in.cuh)](./gpu_cluster/source_code/Data_objects/GPU_grid_in.cuh): 
  - [GPU_grid_out.cuh](./gpu_cluster/source_code/Data_objects/GPU_grid_out.cuh): 
  - [GPU_shared_in.cuh](./gpu_cluster/source_code/Data_objects/GPU_shared_in.cuh):


## CUDA (GPU) 
[config_GPU.h](./gpu_cluster/source_code/config_GPU.h): Constants for GPU 
