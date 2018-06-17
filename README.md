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
The step will generate a new database file with reverse sequences. Update the path of `FASTA_Database` in the configuration file.

#### <a name="labelds"></a>Running GPA

There are two basic versions of GPA: one for running on a single machine and another for running with MPI on a cluster.  

* __Single Machine Version:__ This version of the assembler should be used if you are going to run the database-searching on a single machine with one or more cores. The searching is invoked through a run `Sipros_OpemMP` in `bin` directory. The quick start command as shown below will be used in a batch job submission script or directly typed on the command line terminal.   

```
#!/bin/bash

# Single MS2 file
Sipros_OpemMP -o output_dir -f ms_data -c SiprosConfig.cfg

# Multiple MS2 files in a working directory
Sipros_OpemMP -o output_dir -w workingdirectory -c SiprosConfig.cfg
```
Results (`.Spe2Pep` files) will be saved on the output directory. if you have many configure files, specify `-g`, like `Sipros_OpemMP -o output_dir -w workingdirectory -g configurefiledirectory`. Use `./Sipros_OpemMP -h` for help information. 

* __MPI Version:__ This version of the database-searching should be used if you are going to run on a cluster with MPI support. The run script to invoke `Sipros_MPI` depends on the cluster management and job scheduling system. An example bash script `submit_job.pbs` is provide in `configs` directory.
 
The quick start commands are:
```
### MPI Verion 
Sipros_MPI -o output_dir -w workingdirectory -c SiprosConfig.cfg
```
Results (`.Spe2Pep` files) will be saved on the output directory. if you have many configure files, specify `-g`, like `Sipros_MPI -o output_dir -w workingdirectory -g configurefiledirectory`.


### Miscellany

### Questions?

* [Xuan Guo](mailto:xuan_guo@outlook.com)
