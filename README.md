# Gene based P-value Adaptive Combination Approach (GPA)

GPA can integrate association evidence from summary statistics, either p value or other statistical value, continuous or binary traits, which might come from the same or different studies of traits. To run GPA, you need summary statistic data in a file. Then issue a command such as:

```
gpa_openmp -i test.txt -o pvalue.txt -p 10000
```

### Current Version
* v1.0

### Setup and Installation

#### Basic Dependencies

1. GNU GCC or Intel C++  with C++11 support i.e. gcc4.9+ or above, icpc15.0+ or above.
2. MPI Library with MPI-3 support i.e. OpenMPI 1.8 and above or cray-mpich/7.4.0 and above. By default the mpic++ wrapper is needed. If you are on a Cray cluster and the wrapper is "CC". You will need to edit the compiler.mk file. Uncomment the line "MCC := CC" and comment out "MCC := mpic++".   
 
#### Installation Steps
1. Download the tarball with compiled executables for Linux with GCC 4.9 and above from  [https://github.com/Biocomputing-Research-Group/GPA/releases](https://github.com/Biocomputing-Research-Group/GPA/releases). The code has been tested only on Linux.
2. If you decide to download the source code, use the following commands to build:
  1. OpenMP version "make openmp".
  2. MPI version version "make mpi" 
  3. All the versions can be built with "make all"
If compiled successfully, the required executables will be in `bin` directory. 

#### Prepare summary statistics
A toy example summary statistic data [test.txt](./toy_example/test.txt) is provided in "toy_example".
The format of this input data is as follows:
1st line, gene name
2nd line, Z-statistic
3rd line, covariance matrix

#### <a name="labelds"></a>Running GPA

There are two basic versions of GPA: one for running on a single machine and another for running with MPI on a cluster.  

* __Single Machine Version:__ This version of GPA should be used if you are going to run on a single machine with one or more cores. The quick start command as shown below will be typed on the command line terminal.   

```
#!/bin/bash

gpa_openmp -i test.txt -o pvalue.txt -p 10000
```
Results ('pvalue.txt' files) will be saved on the output directory.
Use `gpa_openmp -h` for help information. 

* __MPI Version:__ This version of GPA should be used if you are going to run on a cluster with MPI support. An example bash script [submit_job.pbs](./toy_example/submit_job.pbs) is provide in "toy_example".
 
The quick start commands are:

```
### MPI Verion 
gpa_mpi -i test.txt -o pvalue.txt -p 10000 -r 10000
```
Results ('pvalue.txt' files) will be saved on the output directory.
Use `gpa_mpi -h` for help information. 

### Miscellany

### Questions?

* [Xuan Guo](mailto:xuan_guo@outlook.com)
