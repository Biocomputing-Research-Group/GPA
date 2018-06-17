/*
 ============================================================================
 Name        : afc.cpp
 Author      : Xuan Guo
 Version     :
 Copyright   : Your copyright notice
 Description : Adaptive Fisher Combination
 ============================================================================
 */
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

#include "gpa_openmp.h"

using namespace std;

/**
 * calculate type 1 error for AFC method
 * -p the number of permutations
 * -i the file containing covariance matrix
 * -o the output file containing p-value
 */
int main(int argc, char *argv[]) {

	main_afc_openmp(argc, argv);

	return 0;
}

