/*
 * afc_mpi.cpp
 *
 *  Created on: Apr 28, 2018
 *      Author: xgo
 */

#include "gpa_mpi.h"

void initialize_pool_mpi(uint64_t _pool_size, uint64_t _dimension, double * _v_Z, gsl_matrix * m_R, bool *& pre_mvrnorm,
		double * & _log_array, uint64_t _permutation) {
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
	_log_array = new double[_permutation + 1];
	_log_array[0] = -log((0.1 / _permutation));
	for (i = 1; i <= _permutation; i++) {
		_log_array[i] = -log(((double) i) / _permutation);
	}

	gsl_vector_free(v_mu);
	gsl_vector_free(v_Z);
}

double master_process(uint64_t _permutation, uint64_t _job_size, uint64_t _dimension, double * v_P) {
	int iNumberOfProcessors, iNumberOfSlaves;
	uint64_t i, j, workloadSize, iBounderOfProcess, currentWorkId, idx;
	MPI_Status status;

	MPI_Comm_size(MPI_COMM_WORLD, &iNumberOfProcessors); /* get number of processes */
	workloadSize = ceil((1.0 * _permutation) / (1.0 * _job_size));
	_permutation = workloadSize * _job_size;
	int count = _job_size * _dimension;

	/* send out the first list of jobs */
	iNumberOfSlaves = iNumberOfProcessors - 1;
	iBounderOfProcess = ((workloadSize <= (uint64_t) iNumberOfSlaves) ? workloadSize : (uint64_t) iNumberOfSlaves);

	for (i = 1; i <= iBounderOfProcess; i++) {
		MPI_Send(&_job_size, 1, MPI_UINT64_T, i, WORKTAG, MPI_COMM_WORLD);
		cout << "Send job " << (i - 1) << endl;
	}

	/* used for save the return from each slave process */
	// vector<double> Tper_slave(_dimension * _job_size);
	double * p_Tper_slave = new double[_dimension * _job_size], *p;

	/* final complete list of all permutation */
	vector<vector<pair<double, uint64_t> > > Tper(_dimension);
	for (i = 0; i < _dimension; ++i) {
		Tper.at(i).resize(_permutation + 2); // the last row is fixed. using maximum value
		Tper.at(i).at(_permutation + 1).first = FLT_MAX;
	}

	/* get the first row for Tper */
	sort(v_P, v_P + _dimension);
	Tper.at(0).at(0).first = -log(v_P[0]);
	Tper.at(0).at(0).second = 0;
	for (j = 1; j < _dimension; ++j) {
		Tper.at(j).at(0).first = Tper.at(j - 1).at(0).first - log(v_P[j]);
		Tper.at(j).at(0).second = 0;
	}
	idx = 1; // next row for saving results from slave processes

	/* distribute the jobs to slaves */
	currentWorkId = iBounderOfProcess;
	while (currentWorkId < workloadSize) {
		MPI_Recv(p_Tper_slave, count, MPI_DOUBLE, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
		cout << "Receive a job result." << endl;
		MPI_Send(&_job_size, 1, MPI_UINT64_T, status.MPI_SOURCE, WORKTAG, MPI_COMM_WORLD);
		cout << "Send job " << currentWorkId << endl;

		p = p_Tper_slave;
		for (i = 0; i < _job_size; i++) {
			for (j = 0; j < _dimension; j++) {
				Tper.at(j).at(idx).first = *p; // p_Tper_slave[i * _dimension + j];
				p++;
				Tper.at(j).at(idx).second = idx;
			}
			idx++;
		}

		currentWorkId++;
	}

	/* collect the results from the first patch */
	p = p_Tper_slave;
	for (uint64_t k = 1; k <= iBounderOfProcess; k++) {
		MPI_Recv(p_Tper_slave, count, MPI_DOUBLE, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
		cout << "Receive a job result." << endl;
		for (i = 0; i < _job_size; i++) {
			for (j = 0; j < _dimension; j++) {
				Tper.at(j).at(idx).first = *p; // p_Tper_slave[i * _dimension + j];
				p++;
				Tper.at(j).at(idx).second = idx;
			}
			idx++;
		}
	}

	/* Tell all the slaves to exit by sending an empty message with the DIETAG. */
	for (i = 1; i <= (uint64_t) iNumberOfSlaves; i++) {
		MPI_Send(0, 0, MPI_INT, i, DIETAG, MPI_COMM_WORLD);
	}

	/* sort Tper */
#pragma omp parallel for \
		    schedule(guided)
	for (i = 0; i < _dimension; i++) {
		sort(Tper.at(i).begin(), Tper.at(i).end());
	}

	vector<double> T2(_permutation + 1, 0);

#pragma omp parallel for \
		    schedule(guided)
	for (i = 0; i <= _permutation; i++) {
		for (uint64_t k = 0; k < _dimension; k++) {
			if (T2[Tper[k][i].second] < i) {
				T2[Tper[k][i].second] = i;
			}
		}
	}

	uint64_t target = 0;

	double sum1 = 0, sum2 = 0;
	for (i = 1; i <= _permutation; i++) {
		if (T2[i] > T2[target]) {
			sum1++;
		} else if (T2[i] == T2[target]) {
			sum2++;
		}
	}

	double pvalue = 0;
	pvalue = sum1 / ((double) _permutation) + sum2 / ((double) _permutation) / 2;

	/* free memory */
	delete[] p_Tper_slave;

	return pvalue;
}

double master_process_equal(uint64_t _permutation, uint64_t _job_size, uint64_t _dimension, double * v_P) {
	int iNumberOfProcessors, iNumberOfSlaves;
	uint64_t i, j, workloadSize, iBounderOfProcess, currentWorkId, idx;
	MPI_Status status;

	MPI_Comm_size(MPI_COMM_WORLD, &iNumberOfProcessors); /* get number of processes */
	workloadSize = ceil((1.0 * _permutation) / (1.0 * _job_size));
	_permutation = workloadSize * _job_size;
	int count = _job_size * _dimension;

	/* send out the first list of jobs */
	iNumberOfSlaves = iNumberOfProcessors - 1;
	iBounderOfProcess = ((workloadSize <= (uint64_t) iNumberOfSlaves) ? workloadSize : (uint64_t) iNumberOfSlaves);

	for (i = 1; i <= iBounderOfProcess; i++) {
		MPI_Send(&_job_size, 1, MPI_UINT64_T, i, WORKTAG, MPI_COMM_WORLD);
		cout << "Send job " << (i - 1) << endl;
	}

	/* used for save the return from each slave process */
	// vector<double> Tper_slave(_dimension * _job_size);
	double * p_Tper_slave = new double[_dimension * _job_size], *p;

	/* final complete list of all permutation */
	vector<vector<pair<double, uint64_t> > > Tper(_dimension);
	for (i = 0; i < _dimension; ++i) {
		Tper.at(i).resize(_permutation + 2); // the last row is fixed. using maximum value
		Tper.at(i).at(_permutation + 1).first = FLT_MAX;
	}

	/* get the first row for Tper */
	sort(v_P, v_P + _dimension);
	Tper.at(0).at(0).first = -log(v_P[0]);
	Tper.at(0).at(0).second = 0;
	for (j = 1; j < _dimension; ++j) {
		Tper.at(j).at(0).first = Tper.at(j - 1).at(0).first - log(v_P[j]);
		Tper.at(j).at(0).second = 0;
	}
	idx = 1; // next row for saving results from slave processes

	/* distribute the jobs to slaves */
	currentWorkId = iBounderOfProcess;
	while (currentWorkId < workloadSize) {
		MPI_Recv(p_Tper_slave, count, MPI_DOUBLE, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
		cout << "Receive a job result." << endl;
		MPI_Send(&_job_size, 1, MPI_UINT64_T, status.MPI_SOURCE, WORKTAG, MPI_COMM_WORLD);
		cout << "Send job " << currentWorkId << endl;

		p = p_Tper_slave;
		for (i = 0; i < _job_size; i++) {
			for (j = 0; j < _dimension; j++) {
				Tper.at(j).at(idx).first = *p; // p_Tper_slave[i * _dimension + j];
				p++;
				Tper.at(j).at(idx).second = idx;
			}
			idx++;
		}

		currentWorkId++;
	}

	/* collect the results from the first patch */
	p = p_Tper_slave;
	for (uint64_t k = 1; k <= iBounderOfProcess; k++) {
		MPI_Recv(p_Tper_slave, count, MPI_DOUBLE, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
		cout << "Receive a job result." << endl;
		for (i = 0; i < _job_size; i++) {
			for (j = 0; j < _dimension; j++) {
				Tper.at(j).at(idx).first = *p; // p_Tper_slave[i * _dimension + j];
				p++;
				Tper.at(j).at(idx).second = idx;
			}
			idx++;
		}
	}

	/* Tell all the slaves to exit by sending an empty message with the DIETAG. */
	for (i = 1; i <= (uint64_t) iNumberOfSlaves; i++) {
		MPI_Send(0, 0, MPI_INT, i, DIETAG, MPI_COMM_WORLD);
	}

	/* sort Tper */
#pragma omp parallel for \
		    schedule(guided)
	for (i = 0; i < _dimension; ++i) {
		sort(Tper.at(i).begin(), Tper.at(i).end());
	}

	/* ranking */
	uint64_t k = 0, l = 0, len = _permutation + 2;
	double * v_rank = new double[_dimension * len];
	double rank_value = 0;
	for (i = 0; i < _dimension; ++i) {
		for (j = 0; j <= _permutation; ++j) {
			rank_value = 0;
			for (k = j; k < len; ++k) {
				if (Tper.at(i).at(j).first != Tper.at(i).at(k).first) {
					break;
				}
				rank_value += k;
			}
			rank_value /= (k - j);
			for (l = j; l < k; ++l) {
				v_rank[i * len + l] = rank_value;
			}
			j = (k - 1);
		}
	}

	/*
	string s_tper = "/media/xgo/workspace/projects/1804_afc/test/tper.txt";
	string s_rank = "/media/xgo/workspace/projects/1804_afc/test/rank.txt";
	ofstream outfile;
	outfile.open(s_tper.c_str());
	if (!outfile.is_open()) {
		cout << "Error: can't create the output file!" << endl;
		return 0;
	}
	for (i = 0; i < len; ++i) {
		for (j = 0; j < _dimension; ++j) {
			outfile << Tper[j][i].first << "\t" << Tper[j][i].second << "\t";
		}
		outfile << endl;
	}
	outfile.close();
	outfile.open(s_rank.c_str());
	if (!outfile.is_open()) {
		cout << "Error: can't create the output file!" << endl;
		return 0;
	}
	for (i = 0; i < len; ++i) {
		for (j = 0; j < _dimension; ++j) {
			outfile << v_rank[j * len + i] << "\t";
		}
		outfile << endl;
	}
	outfile.close();*/

	vector<double> T2(_permutation + 1, 0);

#pragma omp parallel for \
		    schedule(guided)
	for (i = 0; i <= _permutation; ++i) {
		for (uint64_t k = 0; k < _dimension; ++k) {
			if (T2[Tper[k][i].second] < v_rank[k * len + i]) {
				T2[Tper[k][i].second] = v_rank[k * len + i];
			}
		}
	}

	uint64_t target = 0;

	double sum1 = 0, sum2 = 0;
	for (i = 1; i <= _permutation; i++) {
		if (T2[i] > T2[target]) {
			sum1++;
		} else if (T2[i] == T2[target]) {
			sum2++;
		}
	}

	double pvalue = 0;
	pvalue = sum1 / ((double) _permutation) + sum2 / ((double) _permutation) / 2;

	/* free memory */
	delete[] p_Tper_slave;
	delete[] v_rank;

	return pvalue;
}

void slave_process(uint64_t _pool_size, uint64_t _dimension, double * v_Z, uint64_t numper, uint64_t _job_size,
		double * pre_mvrnorm) {
	MPI_Status status;
	uint64_t job_size, i;
	int myid;
	MPI_Comm_rank(MPI_COMM_WORLD, &myid);

	uint64_t num_element = _dimension * _job_size;
	// vector<double> Tper(num_element);
	double * p_Tper = new double[num_element];
	uint64_t nthreads = omp_get_max_threads();
	cout << "Slave " << myid << ": number of threads: " << nthreads << endl;
	vector<double *> v_p_P0(nthreads);
	vector<gsl_rng *> v_r(nthreads);
	for (i = 0; i < nthreads; i++) {
		v_p_P0[i] = new double[_dimension];
		v_r[i] = gsl_rng_alloc(gsl_rng_mt19937);
		unsigned long seed;
		seed = time(NULL);
		gsl_rng_set(v_r[i], seed); // Seed with time
	}

	double min_count = 0.1 / numper;

	while (true) {
		/* receive a job */
		MPI_Recv(&job_size, 1, MPI_UINT64_T, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
		if (status.MPI_TAG == DIETAG) {
			break;
		}

		/* afc part */
#pragma omp parallel for \
	    schedule(guided)
		for (i = 0; i < _job_size; ++i) {
			/* Obtain thread number */
			int tid = omp_get_thread_num();
			/* clean P0 */
			fill(v_p_P0[tid], v_p_P0[tid] + _dimension, 0);
			uint64_t j, k, select = 0;
			double * p, *p_Z;
			for (j = 0; j < numper; ++j) {
				select = gsl_rng_uniform_int(v_r[tid], _pool_size);
				p = &pre_mvrnorm[select * _dimension];
				p_Z = v_Z;
				for (k = 0; k < _dimension; k++) {
					if (*p > *p_Z) {
						v_p_P0[tid][k]++;
					}
					p++;
					p_Z++;
				}
			}

			for (j = 0; j < _dimension; ++j) {
				v_p_P0[tid][j] /= numper;
				if (v_p_P0[tid][j] == 0) {
					v_p_P0[tid][j] = min_count;
				}
			}
			sort(v_p_P0[tid], v_p_P0[tid] + _dimension);
			uint64_t idx = i * _dimension;
			p = &p_Tper[idx];
			*p = -log(v_p_P0[tid][0]);
			for (j = 1; j < _dimension; ++j) {
				*(p + 1) = *p - log(v_p_P0[tid][j]);
				p++;
			}
		}

		/* send results to master */
		MPI_Send(p_Tper, num_element, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
	}

	/*clean memory*/
	delete[] p_Tper;

	for (i = 0; i < nthreads; i++) {
		delete[] v_p_P0[i];
		gsl_rng_free(v_r[i]);
	}

	cout << "Slave " << myid << " is done." << endl;

}

void slave_process_fast(uint64_t _pool_size, uint64_t _dimension, double * v_Z, uint64_t numper, uint64_t _job_size,
		bool * pre_mvrnorm, double * _log_array) {
	MPI_Status status;
	uint64_t job_size, i;
	int myid;
	MPI_Comm_rank(MPI_COMM_WORLD, &myid);

	uint64_t num_element = _dimension * _job_size;
	// size_t len_p_P0 = _dimension * sizeof(uint64_t);
	double * p_Tper = new double[num_element];

	/* multi-thread data structures */
	uint64_t nthreads = omp_get_max_threads();
	cout << "Slave " << myid << ": number of threads: " << nthreads << endl;
	vector<uint64_t *> v_p_P0(nthreads);
	vector<gsl_rng *> v_r(nthreads);
	for (i = 0; i < nthreads; i++) {
		v_p_P0[i] = new uint64_t[_dimension];
		v_r[i] = gsl_rng_alloc(gsl_rng_mt19937);
		unsigned long seed;
		seed = time(NULL);
		gsl_rng_set(v_r[i], seed); // Seed with time
	}

	/* keep receiving jobs */
	while (true) {
		/* receive a job */
		MPI_Recv(&job_size, 1, MPI_UINT64_T, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

		/* die tag, break while */
		if (status.MPI_TAG == DIETAG) {
			break;
		}

		/* afc part */
#pragma omp parallel for \
	    schedule(guided)
		for (i = 0; i < _job_size; ++i) {
			/* Obtain thread number */
			int tid = omp_get_thread_num();
			/* clean P0 */
			fill(v_p_P0[tid], v_p_P0[tid] + _dimension, 0);
			// memset(v_p_P0[tid], 0, len_p_P0);
			uint64_t j, k, i_select = 0;
			bool * p;
			uint64_t * p_P0;
			for (j = 0; j < numper; ++j) {
				i_select = gsl_rng_uniform_int(v_r[tid], _pool_size);
				p = &pre_mvrnorm[i_select * _dimension];
				p_P0 = v_p_P0[tid];
				for (k = 0; k < _dimension; ++k) {
					if (p[k]) {
						p_P0[k]++;
					}
					// *p_P0++ += *p++;
				}
			}

			double * p_T;

			sort(v_p_P0[tid], v_p_P0[tid] + _dimension);
			uint64_t idx = i * _dimension;
			p_T = &p_Tper[idx];
			p_P0 = v_p_P0[tid];
			*p_T = _log_array[*p_P0];
			for (j = 1; j < _dimension; ++j) {
				p_T[j] = p_T[j - 1] + _log_array[p_P0[j]];
			}
		}

		/* send results to master */
		MPI_Send(p_Tper, num_element, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
	}

	/*clean memory*/
	delete[] p_Tper;
	cout << "Slave " << myid << " is done." << endl;

}

int main_afc_mpi(int argc, char *argv[]) {

	MPI_Init(&argc, &argv); /* starts MPI */

	uint64_t i, j, dimension;

	uint64_t numper_permutations = 1000;
	string s_inputfile = "covariancematrix.txt";
	string s_outfile = "result.txt";
	uint64_t factor = 20; // if 40, use 64 GB for 20 SNPs
	uint64_t job_size = 1000;
	uint64_t num_traits = 1; // number of traits, default there is only one trait

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
		} else if (vsArguments[i] == "-f") {
			factor = atoi(vsArguments[++i].c_str());
		} else if (vsArguments[i] == "-j") {
			job_size = atoi(vsArguments[++i].c_str());
		} else if (vsArguments[i] == "-t") {
			num_traits = atoi(vsArguments[++i].c_str());
		} else if ((vsArguments[i] == "-h") || (vsArguments[i] == "--help")) {
			cout << "GPA version 1.0" << endl;
			cout << "Usage: " << endl;
			cout << "-i File containing gene name, Z-statistic, p-value, and covariance matrix" << endl;
			cout << "-o Output file for storing the p-value using GPA" << endl;
			cout << "-p Number of permutations in total" << endl;
			cout << "-j Number of permutations per computing node (default 1000)" << endl;
			cout << "-n Number of tested traits (default 1)" << endl;
			cout << "-t Number of traits (default 1)" << endl;
			cout << "-h Show help message" << endl;
			cout << "Example:" << endl;
			cout << "gpa_mpi -p 10000 -i test.txt -o p-value.txt" << endl;
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

	double * v_Z = new double[dimension];
	double * v_P = new double[dimension];
	for (i = 0; i < dimension; i++) {
		v_Z[i] = vd_Z.at(i);
		v_P[i] = vd_P.at(i);
	}

	double begin = omp_get_wtime();
	int myid;
	MPI_Comm_rank(MPI_COMM_WORLD, &myid); /* get current process id */
	if (myid == 0) {
		cout << "Master starts." << endl;
		ofstream outfile;
		outfile.open(s_outfile.c_str());
		if (!outfile.is_open()) {
			cout << "Error: can't create the output file!" << endl;
			return 0;
		}
		outfile.close();

		double pvalue = master_process(numper_permutations, job_size, dimension, v_P);
		outfile.open(s_outfile.c_str());
		if (outfile.is_open()) {
			outfile << s_gene << "'s p-value:\t" << pvalue << endl;
		} else {
			cout << "Error: can't create the output file!" << endl;
			return 0;
		}
		outfile.close();
		double end = omp_get_wtime();
		cout << "Used\t" << double(end - begin) << "\tSeconds." << endl << endl;
	} else {
		cout << "Slave " << myid << " starts." << endl;

		/*		double * pre_mvrnorm = NULL;
		 initialize_pool_openmp(pool_size, dimension, m_R, pre_mvrnorm);
		 slave_process(pool_size, dimension, v_Z, numper_permutations, job_size, pre_mvrnorm);
		 delete[] pre_mvrnorm;*/

		bool * pre_mvrnorm = NULL;
		double * log_array = NULL;

		initialize_pool_mpi(pool_size, dimension, v_Z, m_R, pre_mvrnorm, log_array, numper_permutations);
		slave_process_fast(pool_size, dimension, v_Z, numper_permutations, job_size, pre_mvrnorm, log_array);

		delete[] pre_mvrnorm;
		delete[] log_array;

	}

	gsl_matrix_free(m_R);
	delete[] v_Z;
	delete[] v_P;
	MPI_Finalize(); /* let MPI finish up ... */
	return 0;
}
