/*
 * afc_openmp.h
 *
 *  Created on: Apr 27, 2018
 *      Author: xgo
 */

#ifndef GPA_OPENMP_H_
#define GPA_OPENMP_H_

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

using namespace std;

// Get the memory usage with a Linux kernel.
inline uint64_t checkMemoryUsage() {
	// get KB memory into count
	uint64_t count = 0;

#if defined(__linux__)
	ifstream f("/proc/meminfo"); // read the linux file
	while (!f.eof()) {
		string key;
		f >> key;
		if (key == "MemTotal:") {     // size of data
			f >> count;
			break;
		}
	}
	f.close();
#endif

	// return MBs memory (size of data)
	return (count / 1024);
}
;

double afc_openmp(uint64_t _pool_size, uint64_t _dimension, double * v_Z, double * v_P, uint64_t numper,
		bool * pre_mvrnorm);
void initialize_pool_openmp(uint64_t _num, uint64_t _dim, double * _v_Z, gsl_matrix * m_R, bool *& pre_mvrnorm);
int main_afc_openmp(int argc, char *argv[]);

int main_test_openmp(int argc, char *argv[]);

#endif /* GPA_OPENMP_H_ */
