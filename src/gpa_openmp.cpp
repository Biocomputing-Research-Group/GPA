/*
 * afc_openmp.cpp
 *
 *  Created on: Apr 27, 2018
 *      Author: xgo
 */

#include "gpa_openmp.h"

double afc_openmp(uint64_t _pool_size, uint64_t _dimension, double * v_Z, double * v_P, uint64_t numper,
		bool * pre_mvrnorm) {

	cout << "GPA is running... " << endl;

	uint64_t inner_per = numper;
	// numper = 30000;

	uint64_t i;
	uint64_t target = 0;

	vector<vector<pair<double, uint64_t> > > Tper(_dimension);
	for (i = 0; i < _dimension; ++i) {
		Tper.at(i).resize(numper + 2); // the last row is fixed. using maximum value
		Tper.at(i).at(numper + 1).first = FLT_MAX;
	}

	sort(v_P, v_P + _dimension);
	Tper.at(0).at(0).first = -log(v_P[0]);
	Tper.at(0).at(0).second = 0;
	for (uint64_t j = 1; j < _dimension; ++j) {
		Tper.at(j).at(0).first = Tper.at(j - 1).at(0).first - log(v_P[j]);
		Tper.at(j).at(0).second = 0;
	}

	uint64_t nthreads = omp_get_max_threads();
	vector<double *> v_p_P0(nthreads);
	vector<gsl_rng *> v_r(nthreads);
	for (i = 0; i < nthreads; i++) {
		v_p_P0[i] = new double[_dimension];
		v_r[i] = gsl_rng_alloc(gsl_rng_mt19937);
		unsigned long seed;
		seed = time(NULL);
		gsl_rng_set(v_r[i], seed); // Seed with time
	}

	double min_count = 0.1 / inner_per;

#pragma omp parallel for \
	    schedule(guided)
	for (i = 1; i <= numper; ++i) {
		/* Obtain thread number */
		int tid = omp_get_thread_num();
		/* clean P0 */
		fill(v_p_P0[tid], v_p_P0[tid] + _dimension, 0);
		uint64_t j, k, select = 0;
		bool * p;
		for (j = 0; j < inner_per; ++j) {
			select = gsl_rng_uniform_int(v_r[tid], _pool_size);
			p = &pre_mvrnorm[select * _dimension];
			for (k = 0; k < _dimension; k++) {
				if (p[k]) {
					v_p_P0[tid][k]++;
				}
			}
		}

		for (j = 0; j < _dimension; ++j) {
			v_p_P0[tid][j] /= inner_per;
			if (v_p_P0[tid][j] == 0) {
				v_p_P0[tid][j] = min_count;
			}
		}
		sort(v_p_P0[tid], v_p_P0[tid] + _dimension);
		Tper.at(0).at(i).first = -log(v_p_P0[tid][0]);
		Tper.at(0).at(i).second = i;
		for (j = 1; j < _dimension; ++j) {
			Tper.at(j).at(i).first = Tper.at(j - 1).at(i).first - log(v_p_P0[tid][j]);
			Tper.at(j).at(i).second = i;
		}
	}

	for (i = 0; i < _dimension; i++) {
		sort(Tper.at(i).begin(), Tper.at(i).end());
	}

	vector<double> T2(numper + 1, 0);

#pragma omp parallel for \
	    schedule(guided)
	for (i = 0; i <= numper; i++) {
		for (uint64_t j = 0; j < _dimension; j++) {
			if (T2[Tper[j][i].second] < i) {
				T2[Tper[j][i].second] = i;
			}
		}
	}

	double sum1 = 0, sum2 = 0;
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

void initialize_pool_openmp(uint64_t _pool_size, uint64_t _dimension, double * _v_Z, gsl_matrix * m_R, bool *& pre_mvrnorm) {

	gsl_rng * local_r = gsl_rng_alloc(gsl_rng_mt19937);
	unsigned long seed;
	seed = time(NULL);
	gsl_rng_set(local_r, seed); // Seed with time

	uint64_t len = _pool_size * _dimension;
	pre_mvrnorm = new bool[len];
	uint64_t i = 0, j = 0;
	gsl_vector * v_mu = gsl_vector_alloc(_dimension);
	gsl_vector_set_zero(v_mu);
	gsl_vector * v_Z = gsl_vector_alloc(_dimension);
	bool * p = pre_mvrnorm;
	double * p_Z;
	for (i = 0; i < _pool_size; i++) {
		gsl_ran_multivariate_gaussian(local_r, v_mu, m_R, v_Z);
		p_Z = _v_Z;
		for (j = 0; j < _dimension; j++) {
			*p = gsl_vector_get(v_Z, j) > *p_Z ? 1 : 0;
			p++;
			p_Z++;
		}
	}

	gsl_vector_free(v_mu);
	gsl_vector_free(v_Z);
	gsl_rng_free(local_r);
}

int main_afc_openmp(int argc, char *argv[]) {
	uint64_t i, j, dimension;

	uint64_t numper_permutations = 1000;
	string s_inputfile = "covariancematrix.txt";
	string s_outfile = "result.txt";
	uint64_t factor = 100; // if 40, use 64 GB for 20 SNPs

	// Grab command line arguments
	vector<string> vsArguments;

	while (argc--) {
		vsArguments.push_back(*argv++);
	}

	for (i = 1; i <= (uint64_t) vsArguments.size() - 1; i++) {
		if (vsArguments[i] == "-o") {
			s_outfile = vsArguments[++i];
		} else if (vsArguments[i] == "-i") {
			s_inputfile = vsArguments[++i];
		} else if (vsArguments[i] == "-p") {
			numper_permutations = atoi(vsArguments[++i].c_str());
		} else if(vsArguments[i] == "-f"){
			factor = atoi(vsArguments[++i].c_str());
		} else if ((vsArguments[i] == "-h") || (vsArguments[i] == "--help")) {
			cout << "GPA version 1.0" << endl;
			cout << "Usage: " << endl;
			cout << "-i File containing gene name, Z-statistic, p-value, and covariance matrix" << endl;
			cout << "-o Output file for storing the p-value using GPA" << endl;
			cout << "-p Number of permutations" << endl;
			cout << "-h Show help message" << endl;
			cout << "Example:" << endl;
			cout << "gpa_openmp -p 10000 -i test.txt -o p-value.txt" << endl;
			exit(0);
		}
	}

	// read in the gene name, z statistic, p value, and covariance matrix
	string s_gene;
	vector<double> vd_Z;
	vector<double> vd_P;
	vector<vector<double> > v_R;
	ifstream inFile(s_inputfile.c_str());
	if (inFile.is_open()) {
		string sline;
		string item;
		// gene name
		getline(inFile, sline);
		s_gene = sline;
		if (s_gene.empty()) {
			cout << "Error: gene name doesn't exist!" << endl;
			return 0;
		}
		// z statistic
		getline(inFile, sline);
		istringstream iss_z(sline);
		for (; iss_z >> item;) {
			vd_Z.push_back(atof(item.c_str()));
		}
		if (vd_Z.empty()) {
			cout << "Error: no Z statistic!" << endl;
			return 0;
		}
		// p value
		getline(inFile, sline);
		istringstream iss_p(sline);
		for (; iss_p >> item;) {
			vd_P.push_back(atof(item.c_str()));
		}
		if (vd_P.empty() || vd_P.size() != vd_Z.size()) {
			cout << "Error: no P-value or P-value doesn't match with Z statistic!" << endl;
			return 0;
		}
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
	} else {
		cout << "Error: the input file doesn't exist!" << endl;
		return 0;
	}
	inFile.close();
	ofstream outfile;
	outfile.open(s_outfile.c_str());
	if (!outfile.is_open()) {
		cout << "Error: can't create the output file!" << endl;
		return 0;
	}
	outfile.close();

	dimension = v_R.size();
	if (dimension != vd_Z.size()) {
		cout << "covariance matrix dimension doesn't match with Z statistic vector!" << endl;
		return 0;
	}
	gsl_matrix * m_R = gsl_matrix_alloc(dimension, dimension);
	for (i = 0; i < dimension; i++) {
		for (j = 0; j < dimension; j++) {
			gsl_matrix_set(m_R, i, j, v_R.at(i).at(j));
		}
	}

	/* Initialize  */

	uint64_t pool_size = factor * numper_permutations;
	uint64_t free_memory = checkMemoryUsage(); // MBs memory
	if (pool_size * dimension / 1024 / 1024 >= free_memory) {
		cout << "Error: memory not large enough for sampling!" << endl;
		return 0;
	}
	bool * pre_mvrnorm = NULL;

	double * v_Z = new double[dimension];
	double * v_P = new double[dimension];
	for (i = 0; i < dimension; i++) {
		v_Z[i] = vd_Z.at(i);
		v_P[i] = vd_P.at(i);
	}

	double begin = omp_get_wtime();

	initialize_pool_openmp(pool_size, dimension, v_Z, m_R, pre_mvrnorm);

	double pvalue = afc_openmp(pool_size, dimension, v_Z, v_P, numper_permutations, pre_mvrnorm);

	outfile.open(s_outfile.c_str());
	if (outfile.is_open()) {
		outfile << s_gene << "'s p-value:\t" << pvalue << endl;
	}else{
		cout << "Error: can't create the output file!" << endl;
		return 0;
	}
	outfile.close();

	gsl_matrix_free(m_R);
	delete[] v_Z;
	delete[] v_P;
	delete[] pre_mvrnorm;
	double end = omp_get_wtime();
	cout << "\n Time used\t" << double(end - begin) << "\tSeconds." << endl << endl;
	return 0;
}
