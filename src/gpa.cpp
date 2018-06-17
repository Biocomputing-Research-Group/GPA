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

#include "gpa_mpi.h"
#include "gpa_openmp.h"
#include "gpa_type1.h"

using namespace std;

gsl_rng * r = gsl_rng_alloc(gsl_rng_mt19937);

double * pre_mvrnorm;

void initialize(unsigned long int _num, unsigned long int _dim, gsl_matrix * m_R) {
	unsigned long int len = _num * _dim;
	pre_mvrnorm = new double[len];
	unsigned long int i = 0, j = 0;
	gsl_vector * v_mu = gsl_vector_alloc(_dim);
	gsl_vector_set_zero(v_mu);
	gsl_vector * v_Z = gsl_vector_alloc(_dim);
	double * p = pre_mvrnorm;
	for (i = 0; i < _num; i++) {
		gsl_ran_multivariate_gaussian(r, v_mu, m_R, v_Z);
		for (j = 0; j < _dim; j++) {
			*p++ = gsl_vector_get(v_Z, j);
		}
	}

	gsl_vector_free(v_mu);
	gsl_vector_free(v_Z);

}

void freememory() {
	delete[] pre_mvrnorm;
}

double afc_type1_boost(unsigned long int _num, unsigned long int _dim, double * v_Z, unsigned long int numper) {

	unsigned long int inner_per = numper;
	numper = 30000;

	unsigned long int i;

	vector<vector<pair<double, unsigned long int> > > Tper(_dim);
	for (i = 0; i < _dim; ++i) {
		Tper.at(i).resize(numper + 2); // the last row is fixed. using maximum value
		Tper.at(i).at(numper + 1).first = FLT_MAX;
	}

	vector<double> v_P0(_dim, 0);
	unsigned long int nthreads = omp_get_max_threads();
	vector<double *> v_p_P0(nthreads);
	vector<gsl_rng *> v_r(nthreads);
	for (i = 0; i < nthreads; i++) {
		v_p_P0[i] = new double[_dim];
		v_r[i] = gsl_rng_alloc(gsl_rng_mt19937);
		unsigned long seed;
		seed = time(NULL);
		gsl_rng_set(v_r[i], seed); // Seed with time
	}

	double min_count = 0.1 / inner_per;

#pragma omp parallel for \
    schedule(guided)
	for (i = 0; i <= numper; ++i) {
		/* Obtain thread number */
		int tid = omp_get_thread_num();
		/* clean P0 */
		fill(v_p_P0[tid], v_p_P0[tid] + _dim, 0);
		unsigned long int j, k, select = 0;
		double * p, *p_Z;
		for (j = 0; j < inner_per; ++j) {
			select = gsl_rng_uniform_int(v_r[tid], _num);
			p = &pre_mvrnorm[select * _dim];
			p_Z = v_Z;
			for (k = 0; k < _dim; k++) {
				if (*p++ > *p_Z++) {
					v_p_P0[tid][k]++;
				}
			}
		}

		for (j = 0; j < _dim; ++j) {
			v_p_P0[tid][j] /= inner_per;
			if (v_p_P0[tid][j] == 0) {
				v_p_P0[tid][j] = min_count;
			}
		}
		sort(v_p_P0[tid], v_p_P0[tid] + _dim);
		Tper.at(0).at(i).first = -log(v_p_P0[tid][0]);
		Tper.at(0).at(i).second = i;
		for (j = 1; j < _dim; ++j) {
			Tper.at(j).at(i).first += Tper.at(j - 1).at(i).first - log(v_p_P0[tid][j]);
			Tper.at(j).at(i).second = i;
		}
	}

	for (i = 0; i < _dim; i++) {
		sort(Tper.at(i).begin(), Tper.at(i).end());
	}

	vector<double> T2(numper + 1, 0);

#pragma omp parallel for \
    schedule(guided)
	for (i = 0; i <= numper; i++) {
		for (unsigned long int j = 0; j < _dim; j++) {
			if (T2[Tper[j][i].second] < i) {
				T2[Tper[j][i].second] = i;
			}
		}
	}

	double sum1 = 0, sum2 = 0;
	unsigned long int target = 0;
	for (i = 1; i <= numper; i++) {
		if (T2[i] > T2[target]) {
			sum1++;
		} else if (T2[i] == T2[target]) {
			sum2++;
		}
	}

	double pvalue = 0;
	pvalue = sum1 / numper + sum2 / numper / 2;

	// cout << sum1 << "\t" << sum2 << "\t" << pvalue << endl;

	// clean the memory
	for (i = 0; i < nthreads; i++) {
		delete[] v_p_P0[i];
		gsl_rng_free(v_r[i]);
	}

	return pvalue;
}

int main_boost(int argc, char *argv[]) {

	unsigned long int i, j, dimension;

	unsigned long int numper_permutations = 1000;
	unsigned long int numper_replica = 1000;
	double alpha = 0.01;
	string s_inputfile = "covariancematrix.txt";
	string s_outfile = "result.txt";

	// Grab command line arguments
	vector<string> vsArguments;

	while (argc--) {
		vsArguments.push_back(*argv++);
	}

	for (i = 1; i <= (unsigned long int) vsArguments.size() - 1; i++) {
		if (vsArguments[i] == "-o") {
			s_outfile = vsArguments[++i];
		} else if (vsArguments[i] == "-i") {
			s_inputfile = vsArguments[++i];
		} else if (vsArguments[i] == "-p") {
			numper_permutations = atoi(vsArguments[++i].c_str());
		} else if (vsArguments[i] == "-r") {
			numper_replica = atoi(vsArguments[++i].c_str());
		} else if (vsArguments[i] == "-a") {
			alpha = atof(vsArguments[++i].c_str());
		}
	}

	unsigned long seed;
	seed = time(NULL);
	gsl_rng_set(r, seed); // Seed with time

	// read in the covariance matrix
	vector<vector<double> > v_R;
	ifstream inFile(s_inputfile.c_str());
	if (inFile.is_open()) {
		string sline;
		string item;
		vector<double> v_tmp;
		while (!inFile.eof()) {
			getline(inFile, sline);
			istringstream iss(sline);
			v_tmp.clear();
			for (; iss >> item;) {
				v_tmp.push_back(atof(item.c_str()));
			}
			if (v_tmp.empty()) {
				break;
			}
			v_R.push_back(v_tmp);
		}
	}
	inFile.close();

	dimension = v_R.size();
	gsl_matrix * m_R = gsl_matrix_alloc(dimension, dimension);
	for (i = 0; i < dimension; i++) {
		for (j = 0; j < dimension; j++) {
			gsl_matrix_set(m_R, i, j, v_R.at(i).at(j));
		}
	}

	/* Initialize  */
	unsigned long int factor = 20; // if 40, use 64 GB for 20 SNPs
	unsigned long int pool_size = factor * numper_permutations;
	initialize(pool_size, dimension, m_R);

	double * v_Z = new double[dimension];
	unsigned long int select = 0;
	double * p, *p_Z;
	vector<double> v_pvalue(numper_replica, 0);
	double type1_error = 0;

	double begin = omp_get_wtime();

	for (i = 0; i < numper_replica; i++) {
		select = gsl_rng_uniform_int(r, pool_size);
		p = &pre_mvrnorm[select * dimension];
		p_Z = v_Z;
		for (j = 0; j < dimension; j++) {
			*p_Z++ = *p++;
		}
		v_pvalue[i] = afc_type1_boost(pool_size, dimension, v_Z, numper_permutations);
		if (v_pvalue[i] < alpha) {
			type1_error++;
		}
		if (i % 10 == 0) {
			cout << "\r" << i << flush; // reprint on the same line
		}
	}

	type1_error /= numper_replica;

	ofstream outfile;
	outfile.open(s_outfile.c_str());
	for (i = 0; i < numper_replica; i++) {
		outfile << v_pvalue[i] << endl;
	}
	outfile.close();

	cout << "\nDone." << endl;
	cout << "Type 1 error:\t" << type1_error << endl;

	gsl_rng_free(r);
	gsl_matrix_free(m_R);
	delete[] v_Z;
	double end = omp_get_wtime();
	cout << "Used\t" << double(end - begin) << "\tSeconds." << endl << endl;
	freememory();
	return 0;
}

int test() {

	int i, j, _dimension = 20, _job_size = 5, _permutation = 10;

	int * p1, *q1;
	int foo1[5] = { 1, 3, 5, 7, 9 };
	int foo2[5] = { 10, 13, 16, 19, 22 };
	p1 = &foo1[0];
	q1 = &foo2[0];
	*p1++ += *q1++;

	// *p1++ = *p1 + *q1++;

	// *p1++ += *p1++ + *q1++;

	bool xyz[20];
	xyz[0] = 1;
	xyz[1] = 1;
	xyz[2] = 0;

	vector<double *> cdf(1);
	cdf[0] = new double[20];

	double * bcd = cdf[0];
	double * acf = & cdf[0][10];

	if (xyz[0]) {
		cout << (100 + xyz[0]) << endl;
		bcd[10]++;
		cout << (*acf) << endl;
		cout << acf[1] << endl;
		bcd[11]+=2;
	}
	if (!xyz[2]) {
		cout << (100 + xyz[2]) << endl;
		cout << bcd[10] << endl;
		cout << *(bcd) << endl;
		cout << acf[1] << endl;
	}

	delete[] cdf[0];
	/* used for save the return from each slave process */
	vector<vector<pair<double, uint64_t> > > Tper_slave(_dimension);
	for (i = 0; i < _dimension; ++i) {
		Tper_slave.at(i).resize(_job_size); // the last row is fixed. using maximum value
		for (j = 0; j < _job_size; j++) {
			Tper_slave.at(i).at(j).first = j;
			Tper_slave.at(i).at(j).second = i;
		}
	}

	/* final complete list of all permutation */
	vector<vector<pair<double, uint64_t> > > Tper(_dimension);
	//vector<pair<double, uint64_t> > * tp;
	for (i = 0; i < _dimension; ++i) {
		Tper.at(i).resize(_permutation + 2); // the last row is fixed. using maximum value
		Tper.at(i).at(_permutation + 1).first = FLT_MAX;

	}

	//tp = & Tper.at(0);

	for (i = 0; i < _dimension; ++i) {
		Tper.at(i) = Tper_slave.at(i);
	}

	Tper.at(0).at(4).first = 10000;

	double * p = new double[20], *x;
	x = p;
	for (int i = 0; i < 20; i++) {
		*x++ = 20 - i;
	}

	sort(p, p + 20);

	fill(p, p + 20, 0);

	p[2]++;

	delete[] p;

	int nthreads, tid;

	cout << omp_get_max_threads() << endl;

	/* Fork a team of threads giving them their own copies of variables */
#pragma omp parallel private(nthreads, tid)
	{

		/* Obtain thread number */
		tid = omp_get_thread_num();
		printf("Hello World from thread = %d\n", tid);

		/* Only master thread does this */
		if (tid == 0) {
			nthreads = omp_get_num_threads();
			printf("Number of threads = %d\n", nthreads);
		}

	} /* All threads join master thread and disband */

#pragma omp parallel for \
    schedule(guided)
	for (i = 0; i < 20; i++) {
		cout << omp_get_thread_num() << ":\t" << i << endl;
	}

	return 0;

}

/**
 * calculate type 1 error for AFC method
 * -p the number of permutations
 * -r the number of replica
 * -i the file containing covariance matrix
 * -o the output file containing p-value
 * -a set the alpha value
 */
int main(int argc, char *argv[]) {

	bool debug = false;
	if (debug) {
		test();
		return 0;
	}

	bool openmp = false;
	if (openmp) {
		main_afc_openmp(argc, argv);
		// main_test(argc, argv);
		return 0;
	}

	bool mpi = true;
	if (mpi) {
		main_afc_mpi(argc, argv);
		return 0;
	}

	bool type1 = false;
	if (type1) {
		main_afc_type1(argc, argv);
		return 0;
	}

	main_boost(argc, argv);

	return 0;
}

