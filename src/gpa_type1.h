/*
 * afc_type1.h
 *
 *  Created on: May 4, 2018
 *      Author: xgo
 */

#ifndef GPA_TYPE1_H_
#define GPA_TYPE1_H_

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

#include <mpi.h>
#include "gpa_openmp.h"

void master_process_type1(uint64_t _num_replica, uint64_t _job_size, string _s_outfile, uint64_t _num_traits);
void slave_process_type1(uint64_t _pool_size, uint64_t _dimension, uint64_t numper, uint64_t _job_size,
		double * pre_mvrnorm, double * _log_array, uint64_t _num_traits, uint64_t _num_SNPs_per_gene);

void initialize_pool_openmp_type1(uint64_t _num, uint64_t _dim, gsl_matrix * m_R, double *& pre_mvrnorm);
void initialize_log_array(uint64_t numper_permutations, double * &log_array);

int main_afc_type1(int argc, char *argv[]);

#endif /* GPA_TYPE1_H_ */
