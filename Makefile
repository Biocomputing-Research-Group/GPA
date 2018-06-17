#############################################
### MAKE file for GPA
#############################################

OpenMP_DIR = ReleaseOpenMP/
MPI_DIR = ReleaseMPI/

openmp:
	$(MAKE) -C $(OpenMP_DIR)
	mkdir -p bin
	cp $(OpenMP_DIR)/gpa_openmp bin/
	
mpi:
	$(MAKE) -C $(MPI_DIR)
	mkdir -p bin
	cp $(MPI_DIR)/gpa_mpi bin/

all:
	$(MAKE) -C $(OpenMP_DIR)
	$(MAKE) -C $(MPI_DIR)
	mkdir -p bin
	cp $(OpenMP_DIR)/gpa_openmp bin/
	cp $(MPI_DIR)/gpa_mpi bin/
	
clean:
	$(MAKE) -C $(OpenMP_DIR) clean
	$(MAKE) -C $(MPI_DIR) clean
	-$(RM) -rf bin/gpa_openmp bin/gpa_mpi
