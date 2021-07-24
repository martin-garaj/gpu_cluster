# gpu_cluster
Boiler-plate framework for job scheduling on HPC GPU cluster. Not every project is finished, but that does not mean the effort is wasted. This project began when Tensorflow and Pytorch were not as popular as they are now. Therefore, there was a plan to use a custom boiler-plate framework to move data within the cluster and do calculations on GPUs. This little project demonstrates, that such a task is doable and be coded in **C++** and **CudaC**.

## Main challenges
  - Moving data between CPU and GPU is solved by using _Unified memory_, which is a physical memory residing on GPU but appears as virtual memory to CPU. This enables to create objects (which inherit from [Managed.cuh](./gpu_cluster/source_code/Data_objects/Managed.cuh)) that are directly constructed and exist only within this memory. This means, that a serialized object sent from _node A_ can be deserialized and directly stored in GPU memory of _node B_. This saves a lot of allocation steps and movind data piece-by-piece.
  - Synchronizing multiple processes residing on unknown nodes. There is a very close co-existence of _PBS Pro_ scheduler and _OpenMPI_ communication interface. While the CPU is not concious of where in the cluster it is, thus neither the _MPI process_, the _PBS Pro_ scheduler can allocate the resources in predictable manner. Then, it is a matter of structuring the _MPI processes_ according to some hierarchical structure using _commRank_ MPI variable.
  - Compilation of code for different architectures. _C++_ and _CudaC_ are different languages that are executed on different machines, therefore they need different compilers. Although this is straight-forward, making the process seamless, thus easy to work with (e.g. compiling on remote machine), was a nice way to practice writting _Makefile_.

# About the project
The project is self-contained, all source-files are provided (including parts of Cereal library) to compile the project. The operating system on cluster is Linux, but the source-code requires very little support other than _Makefile_. 

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

## [CPU](./gpu_cluster/source_code/) (C++)
[main.cpp](./gpu_cluster/source_code/main.cpp): Main file representing _main_ and _worker_ nodes
  - _main_ node is differentiated by [const int] **commRank == ROOT_PROCESS** (defined in [const.h](./gpu_cluster/source_code/const.h))
  - _worker_ nodes are all other nodes with **commRank != ROOT_PROCESS** 
  
[Process.cpp](./gpu_cluster/source_code/Process.cpp), [Process.hpp](./gpu_cluster/source_code/Process.hpp): Object representing both _main_ and _worker_ nodes (NOTICE: the same kind of _Process_ object represents different kinds of nodes)
  - NOTICE: The object has a regular OOP structure:
    - [Process.cpp](./gpu_cluster/source_code/Process.cpp) contains implementations of classes and functions 
    - [Process.hpp](./gpu_cluster/source_code/Process.hpp) contains declarations + implementations of [template functions](https://github.com/martin-garaj/gpu_cluster/blob/45a0ebc99051b16a3dbca8e8fcef00032a10187a/source_code/Process.hpp#L114) (template functions cannot be directly compiled from implementation in .hpp file)


## [Unified memory](./gpu_cluster/source_code/Data_objects/) (CUDA<->C++)
[Managed.cuh](./gpu_cluster/source_code/Data_objects/Managed.cuh): Inheriting from this class allows the object to be _unified memory_ (memory on GPU that is visible from CPU)
  - [GPU_grid_in.cuh)](./gpu_cluster/source_code/Data_objects/GPU_grid_in.cuh): Strucutre sent from _main_ to _worker_ nodes containing data.
  - [GPU_grid_out.cuh](./gpu_cluster/source_code/Data_objects/GPU_grid_out.cuh): Strucutre sent from _workers_ to _main_ node containing results.
  - [GPU_shared_in.cuh](./gpu_cluster/source_code/Data_objects/GPU_shared_in.cuh): Structure shared among _main_ and _workers_ to exchange debug info.


## [GPU](./gpu_cluster/source_code/Cuda/) (CUDA)
[Cuda_GPU.cu](./gpu_cluster/source_code/Cuda/Cuda_GPU.cu): Class representing GPU from the CPU point of view. Therefore, this object, while using CUDA functions and compiled by CUDA compiler, does not run directly on GPU. 
[Cuda_kernel.cu](./gpu_cluster/source_code/Cuda/Cuda_kernel.cu): This file defines _kernel_execute()_ function, which is a wrapper for a function executed on GPU. The actuall kernel function that is exeuted on GPU is implemented in [kernel_by_ref.cu](./gpu_cluster/source_code/Cuda/kernel_by_ref.cu). See [Cuda_kernel.cuh](./gpu_cluster/source_code/Cuda/Cuda_kernel.cuh) for detailed explanation on why C and C++ does not mix. In short, C++ mangles the names during compilation, C does not mangle the names.

## [Utility functions](./gpu_cluster/source_code/Utility/)
  - [Console_print.cpp](./gpu_cluster/source_code/Utility/Console_print.cpp): Pretty console output.
  - [Stop_watch.cpp](gpu_cluster/source_code/Utility/Stop_watch.cpp): Time measurements.
 
## Other files
[return_values.h](./gpu_cluster/source_code/Utility/return_values.h): Define unified return values.
[const.h](./gpu_cluster/source_code/const.h): Constants for CPU.
[config_GPU.h](./gpu_cluster/source_code/config_GPU.h): Constants for GPU.
[Makefile](./gpu_cluster/source_code/Makefile): Makefile to locally (on cluster) compile the source files and generate executable _test\_framework_
[/cereal](./gpu_cluster/source_code/cereal/): Parts of [Cereal library](https://uscilab.github.io/cereal/) required to compile the project.

