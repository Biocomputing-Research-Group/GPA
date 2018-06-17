# GNU Compiler for OpenMP installations
CC := g++
OPTS := -std=c++11 -fopenmp -O3 -g -L/usr/local/lib -I/usr/local/include/

# Intel Compiler for OpenMP installations
# CC := icpc
# OPTS := -std=c++11 -fopenmp -O3 -g

# MPI C++ Compiler for OpenMPI installations (default)
MCC := /usr/local/bin/mpiCC
MOPTS := -std=c++11 -fopenmp -O3 -g -L/usr/local/lib -I/usr/local/include/

# Intel MPI C++ Compiler installations
# MCC := icpc
# MGCC := icc
# MOPTS := -std=c++11 -qopenmp -O3 -g -lmpi -lmpi++
