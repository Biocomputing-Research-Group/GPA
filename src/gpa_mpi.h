/*
 * afc_mpi.h
 *
 *  Created on: Apr 28, 2018
 *      Author: xgo
 */

#ifndef GPA_MPI_H_
#define GPA_MPI_H_

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_matrix.h>

#include <fstream>
#include <vector>
#include <algorithm>    // std::sort
#include <math.h>       /* log */
#include <iostream>
#include <float.h>
#include <time.h>
#include <sstream>
#include <stdint.h>
#include <string>
#include <string.h>

#include <mpi.h>
#include "gpa_openmp.h"

using namespace std;

#define WORKTAG    1
#define DIETAG     2

void initialize_pool_mpi(uint64_t _num, uint64_t _dim, double * _v_Z, gsl_matrix * m_R, bool *& pre_mvrnorm,
		double * & _log_array, uint64_t _permutation);

double master_process(uint64_t _permutation, uint64_t _job_size, uint64_t _dimension, double * v_P);
void slave_process(uint64_t _pool_size, uint64_t _dimension, double * v_Z, uint64_t numper, uint64_t _job_size,
		double * pre_mvrnorm);

void slave_process_fast(uint64_t _pool_size, uint64_t _dimension, double * v_Z, uint64_t numper, uint64_t _job_size,
		bool * pre_mvrnorm, double * _log_array);

int main_afc_mpi(int argc, char *argv[]);

int main_test_mpi(int argc, char *argv[]);

#endif /* GPA_MPI_H_ */
