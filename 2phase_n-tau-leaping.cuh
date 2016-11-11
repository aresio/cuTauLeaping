#ifndef __2PNTAU__
#define __2PNTAU__

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <curand.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <stdio.h>

#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>


#define ACCESS_SAMPLE  larg*DEV_CONST_SAMPLESPECIES*campione + larg*s + tid  

#define DEV_NORMAL_MATRIX 0
#define DEV_FLATTEN_MATRIX 1
#define DEV_FLATTEN_MATRIX_CONST 2
#define DEV_FLATTEN_MATRIX_CONST_GLOBPARAMS 3

#define PI  3.141592653589793238462 

#define USE_NORMAL 0
#define USE_POISSON 1
#define USE_CURAND 2

#define USE_XORWOW 

#define DO_SSA_STEPS
#define DO_TAU_STEPS

#define USE_VOTING 
#define USE_DOUBLE

typedef char  stoich_t;
typedef char   sh_type;
#ifdef USE_DOUBLE 
	typedef double param_t;
	typedef double tau_t;
#else
	typedef float param_t;
	typedef float tau_t;
#endif

typedef unsigned char hor_t;
typedef float g_t;
typedef float musigma_t;
typedef char4 stoich_f;
// typedef char3 stoich_f;
/* struct stt {
	char x;
	char y;
	char z;
};
typedef struct stt stoich_f; */

// extern __shared__ sh_type global_array[];
 
__constant__ tau_t DEV_CONST_TIMEMAX = 0.;
__constant__ unsigned int DEV_CONST_SPECIES = 0;
__constant__ unsigned int DEV_CONST_REACTIONS = 0;

__constant__ unsigned int DEV_CONST_SAMPLES = 0;
__constant__ unsigned int DEV_CONST_SAMPLESPECIES = 0;

// __constant__ unsigned int DEV_CONST_FEEDS = 0;
__constant__ unsigned int DEV_CONST_FEEDSPECIES = 0;

__constant__ unsigned int nc = 10;
__constant__ musigma_t eps = 0.03f;
__constant__ unsigned int APPROX_THRESHOLD = 12;

__constant__ unsigned int DEV_CONST_FLATTEN_LEFT = 0;
__constant__ unsigned int DEV_CONST_FLATTEN_VAR = 0;
__constant__ unsigned int DEV_CONST_FLATTEN_FEAR = 0;

__constant__ double stp = 2.50662827465e0;		// costante utilizzata dalla poissoniana
// __constant__ static double coef[6]={76.18009173e0, -86.50532033e0,   24.01409822e0, -1.231739516e0,  0.120858003e-2, -0.536382e-5}; 

// NEW EXPERIMENTAL
__constant__ stoich_t DEV_CONST_MAT_FEAR[6400] = {0};
__constant__ stoich_f DEV_CONST_MAT_FEARFLAT[500] = {0};
__constant__ stoich_f DEV_CONST_MAT_LEFTFLAT[500] = {0};

__constant__ unsigned int DEV_CONST_EXPERIMENTS = 0;
__constant__ unsigned int DEV_CONST_REPETITIONS = 0;
__constant__ unsigned int DEV_CONST_PAR_REPETITIONS = 0;
__constant__ unsigned int DEV_CONST_TARGET_QUANTITIES = 0;
__constant__ unsigned int DEV_CONST_TIMESLUN = 0;
__constant__ unsigned int DEV_CONST_TOTALTHREADS = 0;

__constant__ float DEV_CONST_ACCUM_NORM = 0;

/* CUDA stuff */
extern stoich_t*		dev_left_matrix;
extern stoich_t*		dev_right_matrix;
extern stoich_t*		dev_var_matrix;
extern stoich_t*		dev_fear_matrix;

extern stoich_f*		dev_left_flatten;
extern stoich_f*		dev_var_flatten;
extern stoich_f*		dev_rav_flatten;
extern stoich_f*		dev_fear_flatten;

extern unsigned int*	dev_qty_matrix;
extern param_t*			dev_par_matrix;
extern hor_t*			dev_hor_vector;
extern hor_t*			dev_hortype_vector;
extern tau_t*			dev_time_vector;
extern unsigned int*	dev_species_to_sample_vector;

extern unsigned int*	dev_perthread_X;
// extern param_t*			dev_perthread_PAR;
extern tau_t*			dev_perthread_T;
extern char*			dev_perthread_SSA;
extern int*				dev_perblock_ended;
extern char*			dev_perthread_CRIT;
extern param_t*			dev_perthread_A0;
extern g_t*				dev_perthread_G;
extern unsigned int *	dev_perthread_storage;
extern unsigned int*	dev_perthread_checkedsample;
// extern tau_t*			dev_perthread_sampletime;
extern unsigned int*	device_target;
extern float*			dev_normalizers;

extern unsigned int*	dev_feed_values;
extern unsigned int*	dev_feed_indeces;

extern char*			thread2experiment ;
extern unsigned int*	global_time_series;

/* DEBUG stuff */
extern unsigned int*	host_X;
extern tau_t*			host_T;
extern param_t*			host_A0;
extern char*			host_SSA;
extern tau_t*			host_sampletime;
extern unsigned int *	host_species_to_sample_vector;
extern char* verifica ;
extern float*			host_normalizers;

// #define _DEBUG

#define DEEP_ERROR_CHECK
#define CUDA_ERROR_CHECK

inline void __cudaCheckError( const char *file, const int line )
{
#ifdef CUDA_ERROR_CHECK

#ifdef DEEP_ERROR_CHECK
	cudaThreadSynchronize();
#endif 

#ifdef _DEBUG
    cudaError err = cudaGetLastError();
    if ( cudaSuccess != err )
    {
        fprintf( stderr, "cudaCheckError() failed at %s:%i : %s\n", file, line, cudaGetErrorString( err ) );
		system("pause");
        exit( -1 );
    }
#endif 

#endif
    return;
}
#define CudaCheckError()    __cudaCheckError( __FILE__, __LINE__ )

void first_pass(std::string folder, unsigned int* species, unsigned int* reactions, bool verbose = false);
void check_args(unsigned a, const char* argv[]);
void read_stoichiometry( std::string folder, bool verbose = false);
void read_feeds( std::string folder, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int*, bool verbose = false);
void read_quantities( std::string folder, unsigned int reactions, unsigned int species, unsigned int parallel_threads, unsigned int* parallel_blocks, bool verbose = false);
void read_quantities_offset( std::string folder, unsigned int reactions, unsigned int parallel_threads, unsigned int pblocks, unsigned int species, unsigned int off_begin, unsigned int off_end , bool verbose = false);
void read_settings( std::string folder, param_t*, unsigned int* );
void read_parameters( std::string folder,  unsigned int GPUs, unsigned int* jobs, unsigned int parallel_threads, unsigned int reactions);
void read_parameters_offset( std::string folder, unsigned int reactions, unsigned int parallel_threads, unsigned int pblocks, unsigned int off_begin, unsigned int off_end, bool verbose = false );
void read_time_max( std::string folder, tau_t* tempo, bool verbose = false );
void set_symbols( unsigned int s, unsigned int r,  tau_t t, unsigned int samples, unsigned int, unsigned int, unsigned int,  bool verbose=false);
void create_rng( unsigned int GPUs, unsigned int* jobs, unsigned int PARALLEL_THREADS, curandState** devStates );
void read_common_parameters( std::string folder, unsigned int reactions);
void read_instants( std::string MODEL_FOLDER, unsigned int *SAMPLES, unsigned int* SAMPLESPECIES , unsigned int, tau_t* tmax, bool verbose = false);
void read_time_series(std::string MODEL_FOLDER,  unsigned int THREADS, unsigned int BLOCKS, unsigned int * REPETITIONS, unsigned int* PARREP, unsigned int* TARGET_QUANTITIES, unsigned int* EXPERIMENTS, unsigned int SAMPLES, unsigned int SAMPLESPECIES, bool dump);

void allocate_space( unsigned int threads, unsigned int blocks, unsigned int species, unsigned int reactions, unsigned int samples, unsigned int GPU, unsigned int, bool verbose=false );
void deallocate_space( unsigned int threads, unsigned int blocks, unsigned int species, unsigned int reactions, unsigned int samples, unsigned int GPU );


void ReadDataBack(unsigned int specie, unsigned int reazioni,unsigned int parallel_threads, unsigned int pblocks, bool dump) ;
void WriteData(float inter, std::string, unsigned int, unsigned int, unsigned int, unsigned int);
void ReadTempi( unsigned int specie, unsigned int reazioni, unsigned int parallel_threads, unsigned int parallel_blocks, bool dump );
void ReadSSA( unsigned int, unsigned int);

__global__ void PrepareVariables( unsigned int* SQ, unsigned int* X,  tau_t* T, unsigned int* checksample, unsigned int * samples, char* SSA, bool FORCE_SSA );

// template<int action>
__global__ void
	__launch_bounds__(64) 
	PreCalculations( 
	unsigned int* X_global,			// S x thread
	param_t* A0_global,				// 1 x thread
	param_t* PAR_global,			// R x thread
	stoich_t* left,
	stoich_f* left_flat,
	stoich_t* var,
	stoich_f* rav,
	hor_t* hor,
	hor_t* hort,
	char* SSA,
	tau_t* tempo,
	unsigned int* prossimo_campione,
	tau_t* samples_times,
	unsigned int * samples_storage,
	unsigned int* dev_species_to_sample_vector, 

	unsigned int* feeds_vector,
	unsigned int* feeds_values,

	g_t* dev_perthread_G,

	
#ifdef USE_XORWOW
	curandState* cs  
#else
	curandStateMRG32k3a* cs
#endif
	, bool FORCE_SSA
	);

/**	Setup CURAND seeds. */
__global__ void setup_random (
	
#ifdef USE_XORWOW
	curandState* cs  
#else
	curandStateMRG32k3a* cs
#endif
	
	, unsigned long seed );

__global__ void EventualSSASteps( 
	const unsigned int steps, 
	unsigned int* X_global, 
	const param_t* PAR_global,
	const stoich_t* left,
	const stoich_f* leftflat,
	const stoich_t* VAR,
	tau_t* tempo, 
	char* SSA,
	unsigned int* prossimo_campione,
	tau_t* samplestimes,
	unsigned int * storage,
	unsigned int* dev_species_to_sample_vector, 

	unsigned int* feeds_vector,
	unsigned int* feeds_values,

#ifdef USE_XORWOW
	curandState* cs  
#else
	curandStateMRG32k3a* cs
#endif
	
	);

__global__ void LightweightSSA( 
	const unsigned int steps, 
	unsigned int* X_global, 
	const param_t* PAR_global,
	const stoich_t* left,
	const stoich_f* leftflat,
	const stoich_t* VAR,
	tau_t* tempo, 
	char* SSA,
	unsigned int* prossimo_campione,
	tau_t* samplestimes,
	unsigned int * storage,
	unsigned int* dev_species_to_sample_vector, 

	unsigned int* feeds_vector,
	unsigned int* feeds_values,

#ifdef USE_XORWOW
	curandState* cs  
#else
	curandStateMRG32k3a* cs
#endif
	
	);


__global__ void ReduceSimulationsCompleted_somma(	char* SSA, char* ended );

/*
template <unsigned int blockSize>
__global__ void reduce6(char *g_idata, int *g_odata, unsigned int n);


template <unsigned int blockSize>
__global__ void
reduce7(char *g_idata, int *g_odata, unsigned int n);
*/

__global__ void ParallelVoting ( tau_t* dev_perthread_T, unsigned int* dev_result_voting );
__global__ void calculateFitness( unsigned int* samples, unsigned int * target, double* fitness, char* swarm, float* normalizers );

// void reduce( dim3 blocksPerGrid, dim3 threadsPerBlock, size_t sharedmem,  char* dev_perthread_SSA, int* dev_perblock_ended, int n);
	
// unsigned int* distribute_work( unsigned int tpb, unsigned int bpt, int* gpus );
void check_shared_memory(unsigned int, unsigned int gpu, bool verbose=false);
void check_constant_memory(unsigned int totale_bytes, bool verbose=false);

void esci();
std::string get_date_dump();
std::string get_output_folder(int, const char*, bool);
bool work_completed(unsigned int, unsigned int, unsigned int GPU );
void start_profiling(cudaEvent_t* start, cudaEvent_t* stop);
float stop_profiling(cudaEvent_t* start, cudaEvent_t* stop);
void WriteDynamics( std::string d, unsigned int parallel_threads, unsigned int pblocks, unsigned int samples, unsigned int species, unsigned int ss );
void WriteDynamicsJustcAMP( std::string d, unsigned int parallel_threads, unsigned int pblocks, unsigned int samples );
void WriteDynamics2( std::string d, unsigned int parallel_threads, unsigned int pblocks, unsigned int samples, unsigned int species, unsigned int samplespecies );
__global__ void test_dump(unsigned int* dev_per_thread_storage, unsigned threads, unsigned blocks, unsigned species, unsigned samples) ;

#endif 
