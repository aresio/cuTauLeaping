
#include "2phase_n-tau-leaping.cuh"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <curand.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
// #include <thrust/reduce.h>
#include <thrust/extrema.h>
// #include <boost\thread.hpp>

#include <stdio.h>
#include <iomanip>
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <ios>
#include <time.h>


/* CUDA stuff */
stoich_t*		dev_left_matrix;
stoich_t*		dev_right_matrix;
stoich_t*		dev_var_matrix;
// stoich_f*		dev_rav_matrix;
stoich_t*		dev_fear_matrix;

stoich_f*		dev_left_flatten;
stoich_f*		dev_var_flatten;
stoich_f*		dev_rav_flatten;
stoich_f*		dev_fear_flatten;

unsigned int*	dev_qty_matrix;
param_t*		dev_par_matrix;
hor_t*			dev_hor_vector;
hor_t*			dev_hortype_vector;
tau_t*			dev_time_vector;
unsigned int*	dev_species_to_sample_vector;

unsigned int*	dev_feed_values;
unsigned int*	dev_feed_indeces;

unsigned int* dev_perthread_X;
// param_t*      dev_perthread_PAR;
tau_t*        dev_perthread_T;
param_t*	  dev_perthread_PROP;
char*		  dev_perthread_SSA;
int*		  dev_perblock_ended;
char*		  dev_perthread_CRIT;
param_t*	  dev_perthread_A0;
g_t*		  dev_perthread_G;
unsigned int* dev_perthread_checkedsample;
unsigned int* dev_perthread_storage;
// tau_t*		  dev_perthread_sampletime;
unsigned int* dev_result_voting;
unsigned int* device_target;
float*		  dev_normalizers;

char*			thread2experiment ;
unsigned int*	global_time_series;


unsigned int*	host_X;
tau_t*			host_T;
param_t*		host_A0;
char*			host_SSA;
tau_t*			host_sampletime;
char*			verifica;
unsigned int*	host_species_to_sample_vector;
float*			host_normalizers;

int main(int argc, const char* argv[])
{
	
	// command line flags defaults
	bool just_fitness = true;
	bool text_fitness = true;
	bool FORCE_SSA    = false;
	bool verbose      = std::string(argv[10])=="1";
	bool stdout_dyn   = std::string(argv[10])=="2";
	unsigned int TOT_SSA_STEPS = 100;


	check_args(argc, argv);
	const char* MODEL_FOLDER = argv[1];
	
	unsigned int PARALLEL_THREADS = atoi(argv[2]);
	unsigned int PARALLEL_BLOCKS = atoi(argv[3]);
	unsigned int GPU = atoi(argv[4]);
	cudaSetDevice(GPU);
	CudaCheckError();
	unsigned int DATA_OFFSET = atoi(argv[5]);
	unsigned int DATA_OFFSET_END = DATA_OFFSET + PARALLEL_BLOCKS*PARALLEL_THREADS ;

	if (verbose) {
		printf(" * Required %d threads per block using %d blocks", PARALLEL_THREADS, PARALLEL_BLOCKS );
		printf(" * Launching simulations on GPU %d \n", GPU );
		printf(" * Data offset: %d to %d threads\n", DATA_OFFSET, DATA_OFFSET_END );
	}

	std::string output_folder = get_output_folder(argc, argv[6], verbose);
	std::string output_prefix(argv[7]);

	if (verbose) printf(" * Output path: %s%s.\n", output_folder.c_str(), output_prefix.c_str());

	if (argc>8){
		just_fitness = std::string(argv[8])!="0";
		text_fitness = std::string(argv[8])=="2";
	}

	if (argc>9) {
		FORCE_SSA = std::string(argv[9])!="0";
		if (FORCE_SSA) { 
			TOT_SSA_STEPS = 100;			
			fprintf(stdout, " * tau-leaping disabled: cuTauLeaping will force SSA instead.\n");		
		}
	}
	
	if (verbose) {
		if ( just_fitness ) {
			printf(" * Fitness calculation: ENABLED \n");	
			if ( text_fitness ) {
				printf(" * Fitness values redirected to stdout.\n");
			}
		} else {
			printf(" * Fitness calculation: DISABLED \n");
		}
	}

	tau_t TIME_MAX = 0;
	unsigned int NUMERO_SPECIE, NUMERO_REAZIONI; 
	unsigned int SAMPLES, SAMPLESPECIES, NUMERO_SPECIE_FEED;
	unsigned int REPETITIONS = 0;
	unsigned int PARREPETITIONS = 0;
	unsigned int TARGET_QUANTITIES = 0;
	unsigned int EXPERIMENTS = 0;

	first_pass( MODEL_FOLDER,  &NUMERO_SPECIE, &NUMERO_REAZIONI);	
	
	read_stoichiometry( MODEL_FOLDER, verbose );
	read_quantities_offset( MODEL_FOLDER, NUMERO_REAZIONI, NUMERO_SPECIE, PARALLEL_THREADS, PARALLEL_BLOCKS, DATA_OFFSET, DATA_OFFSET_END, verbose );
	// read_feeds( MODEL_FOLDER, NUMERO_SPECIE, PARALLEL_THREADS, PARALLEL_BLOCKS, DATA_OFFSET, &NUMERO_SPECIE_FEED ,verbose );	
	read_parameters_offset( MODEL_FOLDER, NUMERO_REAZIONI, PARALLEL_THREADS, PARALLEL_BLOCKS, DATA_OFFSET, DATA_OFFSET_END ,verbose );
	// read_time_max( MODEL_FOLDER, &TIME_MAX ,verbose);
	read_instants( MODEL_FOLDER, &SAMPLES, &SAMPLESPECIES, NUMERO_SPECIE , &TIME_MAX, verbose);	
	if (just_fitness) read_time_series( MODEL_FOLDER,  
		PARALLEL_THREADS, PARALLEL_BLOCKS, 
		&REPETITIONS, &PARREPETITIONS, &TARGET_QUANTITIES, 
		&EXPERIMENTS, SAMPLES, SAMPLESPECIES, verbose );
	allocate_space( PARALLEL_THREADS, PARALLEL_BLOCKS, NUMERO_SPECIE, NUMERO_REAZIONI, SAMPLES, GPU, SAMPLESPECIES, verbose );
		

	dim3 threadsPerBlock( PARALLEL_THREADS, 1,1 );
	dim3 blocksPerGrid( PARALLEL_BLOCKS, 1,1 );

	if (verbose) fprintf(stdout, " * Using %d tpb, %d bpg.\n", PARALLEL_THREADS, PARALLEL_BLOCKS);

	srand( (unsigned long)time(NULL) );
	// srand( 0 );

#ifdef USE_XORWOW
	curandState* devStates;					// random states
	cudaMalloc ( &devStates, PARALLEL_THREADS*PARALLEL_BLOCKS*sizeof( curandState ) );
#else
	curandStateMRG32k3a* devStates;
	cudaMalloc ( &devStates, PARALLEL_THREADS*PARALLEL_BLOCKS*sizeof( curandStateMRG32k3a ) );
#endif
	
	CudaCheckError();

	cudaMalloc( &dev_result_voting, sizeof(unsigned int) );
	CudaCheckError();

	unsigned long seedvalue = (unsigned long)time(NULL);
	if (verbose) printf(" * Seed: %lu\n", seedvalue);

	setup_random <<< blocksPerGrid, threadsPerBlock >>> ( devStates, seedvalue );
	CudaCheckError();
	
	// std::string dataoradump = get_date_dump();
		
	set_symbols( NUMERO_SPECIE, NUMERO_REAZIONI, TIME_MAX, SAMPLES, SAMPLESPECIES, 
		REPETITIONS, EXPERIMENTS,  verbose );
	CudaCheckError();
	
	PrepareVariables<<< blocksPerGrid, threadsPerBlock >>> 
		( dev_qty_matrix, dev_perthread_X, dev_perthread_T, dev_perthread_checkedsample, dev_perthread_storage, dev_perthread_SSA, FORCE_SSA );
	CudaCheckError();

	unsigned int totale_bytes_pre = 
		PARALLEL_THREADS * 
		(NUMERO_SPECIE * sizeof(unsigned int) +			// X
		 NUMERO_REAZIONI * sizeof(param_t) +			// PAR
		 NUMERO_REAZIONI * sizeof(param_t) +			// PROP
		 NUMERO_REAZIONI * sizeof(unsigned int) +		// K-rule
		 NUMERO_SPECIE * sizeof(int) +					// X1
		 NUMERO_REAZIONI * sizeof(char)  				// CRIT		 
		 );
	
	unsigned int totale_bytes_ssa;

	if (!FORCE_SSA) {
		totale_bytes_ssa = 
		PARALLEL_THREADS * 
		(NUMERO_SPECIE * sizeof(unsigned int) +			// X
		 NUMERO_REAZIONI * sizeof(param_t) +			// PAR
		 NUMERO_REAZIONI * sizeof(param_t)				// PROP		 
		 );
	} else{
		totale_bytes_ssa = 
		PARALLEL_THREADS * 
		(NUMERO_SPECIE * sizeof(unsigned int) +			// X
		 NUMERO_REAZIONI * sizeof(param_t)				// PROP		 
		 );
	}

	unsigned int totale_bytes_cnst = 
		NUMERO_SPECIE * NUMERO_REAZIONI * sizeof(stoich_f) +800; // flattened matrices are fixed	
	
	// check the required amount of shared and costant memory 
	// because the model could outisize the resources of GPU
	if   (FORCE_SSA) check_shared_memory(totale_bytes_ssa, GPU, verbose);
	else	         check_shared_memory(totale_bytes_pre, GPU, verbose);
	check_constant_memory(totale_bytes_cnst, verbose);

	// check also global memory usage
	size_t avail, total;
	cudaMemGetInfo( &avail, &total );
	if ( avail==0 ) {
		printf("ERROR: insufficient memory on GPU%d\n", GPU);
		deallocate_space( PARALLEL_THREADS, PARALLEL_BLOCKS, NUMERO_SPECIE, NUMERO_REAZIONI, SAMPLES, GPU );
		exit(-3);
	} 

	if (verbose) fprintf(stdout, " * 3P-tau-leaping starting on GPU%d\n", GPU);

	cudaEvent_t start, stop;
	start_profiling(&start, &stop);

	do {

	
		CudaCheckError();
#ifdef DO_TAU_STEPS 		
		if (!FORCE_SSA) {
			PreCalculations
				<<< blocksPerGrid, threadsPerBlock, totale_bytes_pre >>>
				( dev_perthread_X, dev_perthread_A0, dev_par_matrix, 
				dev_left_matrix, dev_left_flatten, dev_var_matrix, dev_rav_flatten,
				dev_hor_vector, dev_hortype_vector, dev_perthread_SSA, dev_perthread_T,
				dev_perthread_checkedsample, dev_time_vector, dev_perthread_storage, dev_species_to_sample_vector, 			
				dev_feed_indeces, dev_feed_values,
				dev_perthread_G, 			
				devStates, FORCE_SSA );
			CudaCheckError();
			cudaThreadSynchronize();
		}
		
#endif 
	
#ifdef DO_SSA_STEPS
		if (!FORCE_SSA) {
		
			EventualSSASteps
			<<< blocksPerGrid, threadsPerBlock, totale_bytes_ssa >>> 
			( TOT_SSA_STEPS , dev_perthread_X, dev_par_matrix, dev_left_matrix, dev_left_flatten, dev_var_matrix, dev_perthread_T, dev_perthread_SSA, 
			dev_perthread_checkedsample, dev_time_vector, dev_perthread_storage, dev_species_to_sample_vector, 

			dev_feed_indeces, dev_feed_values,

			devStates );
			CudaCheckError();
			cudaThreadSynchronize();
		}
		else {
			LightweightSSA 
			<<< blocksPerGrid, threadsPerBlock, totale_bytes_ssa >>> 
			( TOT_SSA_STEPS , dev_perthread_X, dev_par_matrix, dev_left_matrix, dev_left_flatten, dev_var_matrix, dev_perthread_T, dev_perthread_SSA, 
			dev_perthread_checkedsample, dev_time_vector, dev_perthread_storage, dev_species_to_sample_vector, 

			dev_feed_indeces, dev_feed_values,

			devStates );
			CudaCheckError();
			cudaThreadSynchronize();
		}
		
#endif

 #ifdef USE_VOTING

		// static int hanfinito = 0;

		ParallelVoting<<< blocksPerGrid, threadsPerBlock >>>( dev_perthread_T,  dev_result_voting );
		CudaCheckError();

		unsigned int risultato_voting;
		cudaMemcpy(&risultato_voting, dev_result_voting, sizeof(unsigned int), cudaMemcpyDeviceToHost);
		if ( risultato_voting == PARALLEL_THREADS*PARALLEL_BLOCKS ) break;
		// if ( risultato_voting == DATA_OFFSET_END ) break;
		// printf("%d\n", risultato_voting);

		/*if ( risultato_voting!=hanfinito) {
			hanfinito = risultato_voting;
			printf(" * Finished %d/%d\n", hanfinito, PARALLEL_THREADS*PARALLEL_BLOCKS );
		}
			if (hanfinito==59) {
				ReadTempi( NUMERO_SPECIE, NUMERO_REAZIONI, PARALLEL_THREADS, PARALLEL_BLOCKS, true );
			} */
		
 # else

		// ReadDataBack(NUMERO_SPECIE, NUMERO_REAZIONI, PARALLEL_THREADS, PARALLEL_BLOCKS, true);

		// system("pause");
		// cudaThreadSynchronize();
		// ReadSSA( PARALLEL_THREADS, PARALLEL_BLOCKS);

		/* ReduceSimulationsCompleted_somma
			 <<<blocksPerGrid, threadsPerBlock, PARALLEL_THREADS*sizeof(char) >>> (dev_perthread_SSA, dev_perblock_ended);*/
		 // reduce (blocksPerGrid, threadsPerBlock, PARALLEL_THREADS*sizeof(int), dev_perthread_SSA, dev_perblock_ended, 2*PARALLEL_BLOCKS*PARALLEL_THREADS);


		thrust::device_ptr<char> dptr(dev_perthread_SSA);
		thrust::device_ptr<char> dresptr = thrust::max_element(dptr, dptr+PARALLEL_BLOCKS*PARALLEL_THREADS);
		char max_value = dresptr[0];

		// printf("Reduction: %d\n", max_value);

		if (max_value==-1) break;
 #endif 

		 cudaThreadSynchronize();
		
		
		//  ReadTempi( NUMERO_SPECIE, NUMERO_REAZIONI, PARALLEL_THREADS, PARALLEL_BLOCKS, true );
		// ReadDataBack(NUMERO_SPECIE, NUMERO_REAZIONI, PARALLEL_THREADS, PARALLEL_BLOCKS, true);
		// WriteData( 0.1, output_folder, NUMERO_SPECIE, NUMERO_REAZIONI, PARALLEL_THREADS, PARALLEL_BLOCKS);		
		
		// system("pause");

	// } while ( !work_completed(PARALLEL_THREADS, PARALLEL_BLOCKS, GPU) );
	}while(true);

	if (verbose) float tempo = stop_profiling(&start,&stop);

	// printf("Simulation time: %f\n", tempo);

	std::string dumpfilefinal = output_folder;
	dumpfilefinal.append(output_prefix);

	cudaThreadSynchronize();
	CudaCheckError();
	
	if (just_fitness==false) {
		if (!stdout_dyn) {
			printf (" * Writing dynamics to file %s\n",dumpfilefinal.c_str() );			
			WriteDynamics2(dumpfilefinal, PARALLEL_THREADS, PARALLEL_BLOCKS, SAMPLES, NUMERO_SPECIE, SAMPLESPECIES);
		} else {
			PrintDynamics2(dumpfilefinal, PARALLEL_THREADS, PARALLEL_BLOCKS, SAMPLES, NUMERO_SPECIE, SAMPLESPECIES);
		}

	} else {

		if (verbose) printf (" * Calculating fitnesses.\n");

		double* device_fitness;
		double* host_fitness;
		char* device_swarms;

		host_fitness = (double*) malloc ( sizeof(double) *PARALLEL_THREADS*PARALLEL_BLOCKS);

		cudaMalloc(&device_fitness,sizeof(double)*PARALLEL_THREADS*PARALLEL_BLOCKS);	
		cudaMalloc(&device_swarms, sizeof(char)  *PARALLEL_THREADS*PARALLEL_BLOCKS);	
		CudaCheckError();

		cudaMemcpy(device_swarms, thread2experiment, sizeof(char)*PARALLEL_BLOCKS*PARALLEL_THREADS, cudaMemcpyHostToDevice);
		CudaCheckError();
		
		dim3 BlocksPerGrid(PARALLEL_BLOCKS/PARREPETITIONS,1,1);
		dim3 ThreadsPerBlock(PARALLEL_THREADS,1,1);

		calculateFitness<<<BlocksPerGrid,ThreadsPerBlock>>>( dev_perthread_storage, device_target, device_fitness, device_swarms, dev_normalizers );
		CudaCheckError();

		cudaThreadSynchronize();

		// cudaMemcpy(device_swarms,s2d->thread2experiment,sizeof(char)*s2d->threads, cudaMemcpyDeviceToHost);
		cudaMemcpy(host_fitness,device_fitness,sizeof(double)*PARALLEL_BLOCKS*PARALLEL_THREADS, cudaMemcpyDeviceToHost);
		CudaCheckError();

		if (text_fitness) {

			for (unsigned int sw=0; sw<PARALLEL_BLOCKS/PARREPETITIONS; sw++) {
				for (unsigned int part=0; part<PARALLEL_THREADS; part++) {
					std::cout << host_fitness[sw*PARALLEL_THREADS + part] << "\t";					
				}
				std::cout << "\t";
			}


		} else {

			printf (" * Dumping fitness to file.\n");
		
			std::string outputfile("./");
			outputfile.append("/pref_allfit");

			std::ofstream dump2(outputfile.c_str());
			dump2 << std::fixed << std::showpoint << std::setprecision(20);
			
			for (unsigned int sw=0; sw<PARALLEL_BLOCKS; sw++) {
				for (unsigned int part=0; part<PARALLEL_THREADS; part++) {
					dump2 << host_fitness[sw*PARALLEL_THREADS + part] << "\t";					
				}
				dump2 << "\n";
			}
			dump2.close();

		}

		
	}

	deallocate_space( PARALLEL_THREADS, PARALLEL_BLOCKS, NUMERO_SPECIE, NUMERO_REAZIONI, SAMPLES, GPU );
		
	// std::string twitta;
	//ss.clear();	
	//ss << tempo;
	//twitta.append(" python twit.py ");
	//twitta.append(ss.str());
	
    return 0;
}

