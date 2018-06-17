/*
 * afc_type1.cpp
 *
 *  Created on: May 4, 2018
 *      Author: xgo
 */

#include "gpa_type1.h"
#include "gpa_mpi.h"

void initialize_log_array(uint64_t numper_permutations, double * & _log_array) {
	uint64_t i;
	_log_array = new double[numper_permutations + 1];
	_log_array[0] = -log((0.1 / numper_permutations));
	for (i = 1; i <= numper_permutations; i++) {
		_log_array[i] = -log(((double) i) / numper_permutations);
	}
}

void slave_process_type1(uint64_t _pool_size, uint64_t _dimension, uint64_t numper, uint64_t _job_size,
		double * pre_mvrnorm, double * _log_array) {
	MPI_Status status;
	uint64_t job_size, i;
	int myid;
	MPI_Comm_rank(MPI_COMM_WORLD, &myid);

	/* permutation of smallest Z statistics, T */
	/*
	 vector<vector<pair<double, uint64_t> > > Tper(_dimension);
	 for (i = 0; i < _dimension; ++i) {
	 Tper.at(i).resize(numper + 2); // the last row is fixed. using maximum value
	 Tper.at(i).at(numper + 1).first = FLT_MAX;
	 }*/

	// vector<double> T2(numper + 1, 0);
	// double sum1 = 0, sum2 = 0;
	uint64_t target = 0;
	double * pvalue_array = new double[_job_size];

	/* data structures for each threads */
	uint64_t nthreads = omp_get_max_threads();
	cout << "Slave " << myid << ": number of threads: " << nthreads << endl;
	vector<vector<pair<double, uint64_t> > > * v_Tper = new vector<vector<pair<double, uint64_t> > > [nthreads];
	vector<double> * v_T2 = new vector<double> [nthreads];
	vector<uint64_t *> v_p_P0(nthreads);
	vector<gsl_rng *> v_r(nthreads);
	for (i = 0; i < nthreads; ++i) {
		v_Tper[i] = vector<vector<pair<double, uint64_t> > >(_dimension);
		for (uint64_t j = 0; j < _dimension; ++j) {
			v_Tper[i].at(j).resize(numper + 2); // the last row is fixed. using maximum value
			v_Tper[i].at(j).at(numper + 1).first = FLT_MAX;
		}
		v_T2[i] = vector<double>(numper + 1, 0);
		v_p_P0[i] = new uint64_t[_dimension];
		v_r[i] = gsl_rng_alloc(gsl_rng_mt19937);
		unsigned long seed;
		seed = time(NULL);
		gsl_rng_set(v_r[i], seed); // Seed with time
	}

	/* while receive a job from master process */
	while (true) {
		/* receive a job */
		MPI_Recv(&job_size, 1, MPI_UINT64_T, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
		if (status.MPI_TAG == DIETAG) {
			break;
		}

#pragma omp parallel for \
		    schedule(guided)
		for (i = 0; i < _job_size; ++i) {

			/* get the Z */
			double * v_Z = &pre_mvrnorm[gsl_rng_uniform_int(v_r[0], _pool_size) * _dimension];

			/* Obtain thread number */
			int tid = omp_get_thread_num();
			uint64_t j, k, l, i_select = 0, *p_P0;
			double * p, *p_Z;
			vector<vector<pair<double, uint64_t> > > * p_Tper = &v_Tper[tid];
			/* permutation */
			for (j = 0; j <= numper; ++j) {
				/* clean P0 */
				fill(v_p_P0[tid], v_p_P0[tid] + _dimension, 0);

				for (k = 0; k < numper; ++k) {
					i_select = gsl_rng_uniform_int(v_r[tid], _pool_size);
					p = &pre_mvrnorm[i_select * _dimension];
					p_Z = v_Z;
					p_P0 = v_p_P0[tid];
					for (l = 0; l < _dimension; ++l) {
						if (*p++ > *p_Z++) {
							*p_P0++ += 1;
						} else {
							p_P0++;
						}
					}
				}

				sort(v_p_P0[tid], v_p_P0[tid] + _dimension);

				p_P0 = v_p_P0[tid];
				p_Tper->at(0).at(j).first = _log_array[*p_P0];
				p_Tper->at(0).at(j).second = j;
				for (l = 1; l < _dimension; ++l) {
					p_Tper->at(l).at(j).first = p_Tper->at(l - 1).at(j).first + _log_array[*++p_P0];
					p_Tper->at(l).at(j).second = j;
				}
			}

			/* calculate a p-value */
			for (j = 0; j < _dimension; ++j) {
				sort(p_Tper->at(j).begin(), p_Tper->at(j).end());
			}

			vector<double> * p_T2 = &v_T2[tid];

			fill(p_T2->begin(), p_T2->end(), 0);

			for (j = 0; j <= numper; ++j) {
				for (k = 0; k < _dimension; ++k) {
					if ((*p_T2)[(*p_Tper)[k][j].second] < j) {
						(*p_T2)[(*p_Tper)[k][j].second] = j;
					}
				}
			}

			double sum1 = 0, sum2 = 0;
			for (j = 1; j <= numper; ++j) {
				if ((*p_T2)[j] > (*p_T2)[target]) {
					sum1++;
				} else if ((*p_T2)[j] == (*p_T2)[target]) {
					sum2++;
				}
			}

			pvalue_array[i] = sum1 / numper + sum2 / numper / 2;

		}

		/* send results to master */
		MPI_Send(pvalue_array, _job_size, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
		// break;
	}

	/*clean memory*/
	delete[] pvalue_array;
	delete[] v_Tper;
	delete[] v_T2;
	for (i = 0; i < nthreads; i++) {
		delete[] v_p_P0[i];
		gsl_rng_free(v_r[i]);
	}

	/* slave process is ended */
	cout << "Slave " << myid << " is done." << endl;
}

void master_process_type1(uint64_t _num_replica, uint64_t _job_size, uint64_t _dimension, double * v_P,
		string _s_outfile) {

	int iNumberOfProcessors, iNumberOfSlaves;
	uint64_t i, workloadSize, iBounderOfProcess, currentWorkId;
	ofstream outfile;
	outfile.open(_s_outfile.c_str());
	if (!outfile.is_open()) {
		cout << "Error: can't create the output file!" << endl;
		exit(1);
	}
	MPI_Status status;

	MPI_Comm_size(MPI_COMM_WORLD, &iNumberOfProcessors); /* get number of processes */
	workloadSize = ceil((1.0 * _num_replica) / (1.0 * _job_size));
	_num_replica = workloadSize * _job_size;
	int count = _job_size;

	/* send out the first list of jobs */
	iNumberOfSlaves = iNumberOfProcessors - 1;
	iBounderOfProcess = ((workloadSize <= (uint64_t) iNumberOfSlaves) ? workloadSize : (uint64_t) iNumberOfSlaves);

	for (i = 1; i <= iBounderOfProcess; i++) {
		MPI_Send(&_job_size, 1, MPI_UINT64_T, i, WORKTAG, MPI_COMM_WORLD);
		cout << "Send job " << (i - 1) << endl;
	}

	/* used for save the return from each slave process */
	double * p_pvalue_slave = new double[_job_size], *p;

	/* distribute the jobs to slaves */
	currentWorkId = iBounderOfProcess;
	while (currentWorkId < workloadSize) {
		MPI_Recv(p_pvalue_slave, count, MPI_DOUBLE, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
		cout << "Receive a job result." << endl;
		MPI_Send(&_job_size, 1, MPI_UINT64_T, status.MPI_SOURCE, WORKTAG, MPI_COMM_WORLD);
		cout << "Send job " << currentWorkId << endl;

		p = p_pvalue_slave;
		for (i = 0; i < _job_size; i++) {
			outfile << (*p) << endl;
			p++;
		}

		currentWorkId++;
	}

	/* collect the results from the first patch */
	for (uint64_t k = 1; k <= iBounderOfProcess; k++) {
		MPI_Recv(p_pvalue_slave, count, MPI_DOUBLE, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
		cout << "Receive a job result." << endl;
		p = p_pvalue_slave;
		for (i = 0; i < _job_size; i++) {
			outfile << (*p) << endl;
			p++;
		}
	}

	/* Tell all the slaves to exit by sending an empty message with the DIETAG. */
	for (i = 1; i <= (uint64_t) iNumberOfSlaves; i++) {
		MPI_Send(0, 0, MPI_INT, i, DIETAG, MPI_COMM_WORLD);
	}

	/* free memory */
	delete[] p_pvalue_slave;
}

void initialize_pool_openmp_type1(uint64_t _pool_size, uint64_t _dimension, gsl_matrix * m_R, double *& pre_mvrnorm) {

	gsl_rng * local_r = gsl_rng_alloc(gsl_rng_mt19937);
	unsigned long seed;
	seed = time(NULL);
	gsl_rng_set(local_r, seed); // Seed with time

	uint64_t len = _pool_size * _dimension;
	pre_mvrnorm = new double[len];
	uint64_t i = 0, j = 0;
	gsl_vector * v_mu = gsl_vector_alloc(_dimension);
	gsl_vector_set_zero(v_mu);
	gsl_vector * v_Z = gsl_vector_alloc(_dimension);
	double * p = pre_mvrnorm;
	for (i = 0; i < _pool_size; i++) {
		gsl_ran_multivariate_gaussian(local_r, v_mu, m_R, v_Z);
		for (j = 0; j < _dimension; j++) {
			*p = gsl_vector_get(v_Z, j);
			p++;
		}
	}

	gsl_vector_free(v_mu);
	gsl_vector_free(v_Z);
	gsl_rng_free(local_r);
}

int main_afc_type1(int argc, char *argv[]) {

	MPI_Init(&argc, &argv); /* starts MPI */

	uint64_t i, j, dimension;

	uint64_t numper_permutations = 1000;
	uint64_t num_replica = 1000;
	string s_inputfile = "covariancematrix.txt";
	string s_outfile = "result.txt";
	uint64_t factor = 20; // if 40, use 64 GB for 20 SNPs
	uint64_t job_size = 1000;

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
		} else if (vsArguments[i] == "-a") {
			num_replica = atoi(vsArguments[++i].c_str());
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

		/* check the result file can be created or not */
		ofstream outfile;
		outfile.open(s_outfile.c_str());
		if (!outfile.is_open()) {
			cout << "Error: can't create the output file!" << endl;
			return 0;
		}
		outfile.close();

		/* distribute jobs, collect p-value, and write results to the output */
		master_process_type1(num_replica, job_size, dimension, v_P, s_outfile);

		/* show the time usage */
		double end = omp_get_wtime();
		cout << "Used\t" << double(end - begin) << "\tSeconds." << endl << endl;

	} else {
		cout << "Slave " << myid << " starts." << endl;

		double * pre_mvrnorm = NULL;
		double * log_array = NULL;
		initialize_pool_openmp_type1(pool_size, dimension, m_R, pre_mvrnorm);
		initialize_log_array(numper_permutations, log_array);
		slave_process_type1(pool_size, dimension, numper_permutations, job_size, pre_mvrnorm, log_array);
		delete[] pre_mvrnorm;
		delete[] log_array;

		/*double end = omp_get_wtime();
		cout << "Used\t" << double(end - begin) << "\tSeconds." << endl << endl;*/
	}

	gsl_matrix_free(m_R);
	delete[] v_Z;
	delete[] v_P;
	MPI_Finalize(); /* let MPI finish up ... */
	return 0;
}

