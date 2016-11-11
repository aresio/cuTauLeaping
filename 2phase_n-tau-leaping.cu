#include "2phase_n-tau-leaping.cuh"

#include "device_launch_parameters.h"
#include <curand.h>
#include <cuda_runtime.h>
#include <float.h>
#include <curand_kernel.h>
#include <math_constants.h>
#include <algorithm>
// #include <thrust\reduce.h>

#include <stdio.h>
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#ifdef _WIN32
#include <direct.h>
#endif

#include <sys/stat.h> 
#include <sys/types.h> 
#include <fcntl.h> 
#ifndef _WIN32
#include <unistd.h>
#endif

#include <cassert>

#define COLLECT_DYNAMICS
#define SERVICEDUMP
#define DISABLE_FEED
#define USE_RAV4
// #define FAKE_FEED
#define USE_CONSTANT_FEAR
#define USE_FEAR_SSA

// #define FORCE_SSA

#define SHARED_R r*bdim+tid
#define SHARED_S s*bdim+tid
#define GLOBAL_R larg*r + gid
#define GLOBAL_S larg*s + gid
#define GLOBAL_SAMPLES larg*DEV_CONST_SPECIES*i + larg*s + gid

#define ACCESS_SPECIES			(parallel_threads*pblocks)*s + (parallel_threads*i) + t
#define ACCESS_FEED_SPECIES		(parallel_threads*pblocks)*pos_m + (parallel_threads*i) + t
#define ACCESS_REACTIONS		(parallel_threads*pblocks)*r + (parallel_threads*i) + t
#define ACCESS_SINGLE			(i*parallel_threads)+t


// #define DEBUG_SSA

/* controlla argomenti */
void check_args(unsigned a, const char* argv[]) {
	if (a<6) {
		printf("ERROR: please specify:\n");
		printf("                       the folder which contains the model\n");
		printf("                       the number of parallel threads\n");
		printf("                       the number of blocks\n");
		printf("                       the device's number\n");
		printf("                       the offset of data for this GPU\n");
		printf("                       the output folder\n");
		printf("                       the prefix for the output files\n");
		printf("                       the type of fitness (0 to disable)\n");
		printf("       for instance: cuTauLeaping ./input 128 4 1 0 output prefix 0\n");
		// system("pause");
		exit(-1);
	}

	// printf(" * Model's input directory: %s\n", argv[1]);
	return;
}


void tokenize( std::string s, std::vector<std::string>& v )  {
    std::istringstream buf(s.c_str());
    for(std::string token; getline(buf, token, '\t'); )
        v.push_back(token);
}



/* Since we have to subdivide the threads on multiple GPUs,
we have to pre-parse  the input files in order to determine
the number of blocks for each GPU and to allocate the correct
input data on the respective global memories. */
void first_pass(std::string folder, unsigned int* species, unsigned int* reactions, bool verbose){

	// step 1: open left matrix
	std::string left_path = folder+"/left_side";
	if (verbose) std::cout << " * Gathering info from left matrix: " << left_path << "\n";
	std::string s, line, token;
	std::vector<std::string> vet1, vet2;
	std::ifstream in(left_path.c_str());
	std::stringstream ss;
	
	if ( in.is_open() ) {

		// brief preprocessing to determine the dimensions
		vet1.clear();
		while(getline(in,line)) { 
			vet1.push_back(line);
		}
		if (verbose) printf(" * Found %d reactions\n", vet1.size());
		*reactions = vet1.size();
	}
	in.close();

	ss << vet1[0];
	vet2.clear();
	while ( std::getline(ss, token, '\t') ) {
		vet2.push_back(token);
	}
	if (verbose) printf(" * Found %d species\n", vet2.size());
	*species = vet2.size();

}

void read_stoichiometry( std::string folder, bool verbose  ) {

	// step 1: open left matrix
	std::string left_path = folder+"/left_side";
	std::string right_path = folder+"/right_side";
	std::string lock_path = folder+"/M_feed";
	std::string s, line, token;
	std::vector<std::string> vet1, vet2;
	
	std::stringstream ss;

	stoich_t* left_matrix;
	stoich_t* right_matrix;
	stoich_t* var_matrix;
	stoich_t* fear_matrix;

	stoich_f* flattened_matrix_left;	unsigned int totale_non_zero_left = 0;
	stoich_f* flattened_matrix_var;		unsigned int totale_non_zero_var = 0;
	stoich_f* flattened_matrix_rav;	
	stoich_f* flattened_matrix_fear;	unsigned int totale_non_zero_fear = 0;


	/* NEW!! parso il vettore delle specie lockate */	
	std::ifstream infear(lock_path.c_str());
	std::vector<unsigned int> locked_species;

	if ( infear.is_open() ) {
				
		getline(infear, line);
		ss << line;
		vet2.clear();
		unsigned int n=0; 
		while ( std::getline(ss, token, '\t') ) {
			unsigned int temp = atoi(token.c_str());
			if (temp!=0) locked_species.push_back(n);
			n++;
		}		
		
	} else {
		// se il file non esiste, non ci sono specie in feed.
		if (verbose) printf("WARNING: vector of feed species ML_feed is not specified\n");
	}
		
	if (verbose) printf(" * %d species in feed detected\n", locked_species.size());
	

	std::ifstream in(left_path.c_str());
	unsigned int reactions, species;
	if ( in.is_open() ) {

		// brief preprocessing to determine the dimensions
		vet1.clear();
		while(getline(in,line)) { 
			vet1.push_back(line);
		}
		reactions = vet1.size();
		
		std::stringstream ss;
		ss << vet1[0];
		vet2.clear();
		while ( std::getline(ss, token, '\t') ) {
			vet2.push_back(token);
		}		
		species = vet2.size();
				
		left_matrix =   (stoich_t*) malloc ( sizeof(stoich_t) * species * reactions );
		right_matrix =  (stoich_t*) malloc ( sizeof(stoich_t) * species * reactions );
		var_matrix =    (stoich_t*) malloc ( sizeof(stoich_t) * species * reactions );
		fear_matrix =   (stoich_t*) malloc ( sizeof(stoich_t) * species * reactions );

		memset( fear_matrix, 0,  sizeof(stoich_t) * species * reactions );

		// populate left matrix
		for (unsigned int i=0; i<reactions; i++) {
			ss.clear();
			ss << vet1[i];
			unsigned int p=0;
			while ( std::getline(ss, token, '\t') ) {
				// convert token in char				
				int res;
				std::stringstream convert(token);
				convert >> res;
				left_matrix[i* species + p] =res;
				if (left_matrix[i*species+p]!=0)	totale_non_zero_left++;				
				p++;
			}
		}
	
	} else {
		perror("ERROR reading left matrix");
		// system("pause");
		exit(-1);
	}

	in.close();

	// creo matrice flatten sinistra
	flattened_matrix_left = (stoich_f*) malloc ( sizeof(stoich_f) * totale_non_zero_left );
	unsigned int n=0;
	for (unsigned int r=0; r<reactions; r++ ) {
		for (unsigned int s=0; s<species; s++) {
			if (left_matrix[r*species+s]!=0) {
				stoich_f u;
				u.x=r;
				u.y=s;
				u.z=left_matrix[r*species+s];
				flattened_matrix_left[n]=u;
				n++;
			}
		}
	}

	in.open( right_path.c_str() );

	if ( in.is_open() ) {

		// brief preprocessing to determine the dimensions
		vet1.clear();
		while(getline(in,line)) { 
			vet1.push_back(line);
		}

		if ( vet1.size() != reactions ) {
			printf("WARNING: left and right matrices have different size, aborting");
			exit(-1);
		}

		// populate right matrix
		for (unsigned int i=0; i<reactions; i++) {
			ss.clear();
			ss << vet1[i];
			unsigned int p=0;
			while ( std::getline(ss, token, '\t') ) {
				// convert token in char				
				int res;
				std::stringstream convert(token);
				convert >> res;
				right_matrix[i* species + p] = res;
				p++;
			}
		}

	} else {
		perror("ERROR reading right matrix");
		// system("pause");
		exit(-1);
	}

	for ( unsigned int r=0; r< reactions; r++ ) {
		for ( unsigned int c=0; c< species; c++ ) {
			var_matrix[r* species + c] = right_matrix[ r* species + c ] - left_matrix[ r* species + c ];

			// se la specie non è in feed, aggiorno la stechiometria
			if ( std::find(locked_species.begin(), locked_species.end(), c) == locked_species.end() ) {
				fear_matrix[r* species + c] = right_matrix[ r* species + c ] - left_matrix[ r* species + c ];	
				// printf("%d, %d OK\n", r, c);
			} else {
				fear_matrix[r* species + c] = 0;
				//printf("%d, %d NO\n", r, c);
			}
			
			if (var_matrix[r*species+c]!=0)		totale_non_zero_var++;
			if (fear_matrix[r*species+c]!=0)	totale_non_zero_fear++;
		}
	}

	flattened_matrix_var = (stoich_f*) malloc ( sizeof(stoich_f) * totale_non_zero_var );
	flattened_matrix_rav = (stoich_f*) malloc ( sizeof(stoich_f) * totale_non_zero_var );
	flattened_matrix_fear = (stoich_f*) malloc ( sizeof(stoich_f) * totale_non_zero_fear );
	
	// creo matrice compressa var
	n=0;
	for (unsigned int r=0; r<reactions; r++ ) {
		for (unsigned int s=0; s<species; s++) {
			if (var_matrix[r*species+s]!=0) {
				stoich_f u;
				u.x=r;
				u.y=s;
				u.z=var_matrix[r*species+s];
				flattened_matrix_var[n]=u;
				n++;
			}
		}
	}

	// creo matrice compressa rav (var sulle colonne)
	n=0;
	for (unsigned int s=0; s<species; s++) {
		for (unsigned int r=0; r<reactions; r++ ) {
			if (var_matrix[r*species+s]!=0) {
				stoich_f u;
				u.x=r;
				u.y=s;
				u.z=var_matrix[r*species+s];
				flattened_matrix_rav[n]=u;
				n++;
			}
		}
	}

	// creo matrice compressa fear
	n=0;
	for (unsigned int r=0; r<reactions; r++ ) {
		for (unsigned int s=0; s<species; s++) {
			if (fear_matrix[r*species+s]!=0) {
				stoich_f u;
				u.x=r;
				u.y=s;
				u.z=fear_matrix[r*species+s];				
				flattened_matrix_fear[n]=u;
				n++;
			}
		}
	}


	// finally, we load everything on the GPU
	size_t totalbytes = sizeof(stoich_t)* species * reactions ;

	cudaMalloc( &dev_left_matrix,  totalbytes );
	cudaMalloc( &dev_right_matrix, totalbytes );
	cudaMalloc( &dev_var_matrix,   totalbytes );
	// cudaMalloc( &dev_fear_matrix,  totalbytes );
	CudaCheckError();

	cudaMalloc( &dev_left_flatten, totale_non_zero_left * sizeof(stoich_f) );
	cudaMalloc( &dev_var_flatten,  totale_non_zero_var  * sizeof(stoich_f) ); 
	cudaMalloc( &dev_rav_flatten,  totale_non_zero_var  * sizeof(stoich_f) ); 
	// cudaMalloc( &dev_fear_flatten, totale_non_zero_fear * sizeof(stoich_f) ); 
	CudaCheckError();

	cudaMemcpy( dev_left_matrix,  left_matrix,  totalbytes, cudaMemcpyHostToDevice );
	cudaMemcpy( dev_right_matrix, right_matrix, totalbytes, cudaMemcpyHostToDevice );
	cudaMemcpy( dev_var_matrix,   var_matrix,   totalbytes, cudaMemcpyHostToDevice );
	// cudaMemcpy( dev_fear_matrix,  fear_matrix,  totalbytes, cudaMemcpyHostToDevice );
	CudaCheckError();

	cudaMemcpy( dev_left_flatten, flattened_matrix_left, totale_non_zero_left * sizeof(stoich_f), cudaMemcpyHostToDevice );
	cudaMemcpy( dev_var_flatten,  flattened_matrix_var,  totale_non_zero_var * sizeof(stoich_f), cudaMemcpyHostToDevice );
	cudaMemcpy( dev_rav_flatten,  flattened_matrix_rav,  totale_non_zero_var * sizeof(stoich_f), cudaMemcpyHostToDevice );
	// cudaMemcpy( dev_fear_flatten, flattened_matrix_fear, totale_non_zero_fear * sizeof(stoich_f), cudaMemcpyHostToDevice );

	CudaCheckError();

	/*printf( " * Test pre fear: ");
	for (unsigned int i=0; i<totale_non_zero_fear; i++) 
		printf( "%d\t", flattened_matrix_fear[i].z);
	printf("\n");*/

	cudaMemcpyToSymbol( DEV_CONST_FLATTEN_LEFT,  &totale_non_zero_left,  sizeof(totale_non_zero_left));
	cudaMemcpyToSymbol( DEV_CONST_FLATTEN_VAR,   &totale_non_zero_var,   sizeof(totale_non_zero_var));
	cudaMemcpyToSymbol( DEV_CONST_FLATTEN_FEAR,  &totale_non_zero_fear,  sizeof(totale_non_zero_fear));	
	

	// load fear matrices
	cudaMemcpyToSymbol( DEV_CONST_MAT_FEARFLAT,  flattened_matrix_fear,  totale_non_zero_fear * sizeof(stoich_f));
	cudaMemcpyToSymbol( DEV_CONST_MAT_FEAR,      fear_matrix,			 totalbytes);	
	cudaMemcpyToSymbol( DEV_CONST_MAT_LEFTFLAT,  flattened_matrix_left,  totale_non_zero_left * sizeof(stoich_f));
	CudaCheckError();

	unsigned int tnzl_back = 0;unsigned int tnzv_back = 0;unsigned int tnzf_back = 0;
	cudaMemcpyFromSymbol( &tnzl_back , DEV_CONST_FLATTEN_LEFT, sizeof(tnzl_back) );
	if (verbose) printf( " * Verified on GPU: flattened left matrix has %d non zero values\n", tnzl_back);
	cudaMemcpyFromSymbol( &tnzv_back , DEV_CONST_FLATTEN_VAR,  sizeof(tnzv_back) );
	if (verbose) printf( " * Verified on GPU: flattened var matrix has %d non zero values\n", tnzv_back);
	/* cudaMemcpyFromSymbol( &tnzv_back , DEV_CONST_FLATTEN_VAR,  sizeof(tnzv_back) );
	printf( " * Verified on GPU: flattened var matrix has %d non zero values\n", tnzv_back); */
	cudaMemcpyFromSymbol( &tnzf_back , DEV_CONST_FLATTEN_FEAR, sizeof(tnzf_back) );
	if (verbose) printf( " * Verified on GPU: flattened fear matrix has %d non zero values\n", tnzf_back);

	/*stoich_t* ritornamento = (stoich_t*) malloc( sizeof(stoich_t) * reactions * species );

	
	cudaMemcpyFromSymbol( &ritornamento , DEV_CONST_MAT_FEAR, totalbytes );
	printf( " * Test post fear: ");
	for (unsigned int i=0; i<species*reactions; i++) 
		printf( "%d\t", ritornamento[i]);
	printf("\n");
	*/

	///// CALCOLA HOR /////
	hor_t* host_vettore_ordine = (hor_t*) malloc ( sizeof(hor_t)* reactions );
	hor_t* host_vettore_hor    = (hor_t*) malloc ( sizeof(hor_t)* species );
	hor_t* host_vettore_hortype= (hor_t*) malloc ( sizeof(hor_t)* species );
	memset( host_vettore_ordine, 0, sizeof(hor_t)* reactions);
	memset( host_vettore_hor, 0, sizeof(hor_t)* species);
	memset( host_vettore_hortype, 0, sizeof(hor_t)* species);

	/// le copio nella mia struttura d'appoggio...
	for (unsigned int r=0; r<reactions; r++ ) {
		hor_t tot = 0;
		for (unsigned int c=0; c<species; c++) {
			tot += left_matrix[r*species+c];
		}
		host_vettore_ordine[r] = tot;
	}

	/// seconda passata per calcolare le HOR
	for (unsigned int c=0; c<species; c++) {
		for (unsigned int r=0; r<reactions; r++) {
			if ( left_matrix[r* species+c] > 0 ) 
				if ( host_vettore_ordine[r] > host_vettore_hor[c] ) {
					host_vettore_hor[c] = host_vettore_ordine[r];	
					host_vettore_hortype[c] = left_matrix[r* species+c];
				}				
				// the higher, the cooler
				if ( host_vettore_hortype[c] < left_matrix[r* species+c] ) {
					host_vettore_hortype[c] = left_matrix[r* species+c];
				}
		}
	}

	if (verbose) printf(" * HOR and HORtype calculated\n");

	totalbytes = sizeof(hor_t)* species;

	cudaMalloc( &dev_hor_vector,   totalbytes );
	cudaMalloc( &dev_hortype_vector, totalbytes );
	CudaCheckError();

	cudaMemcpy( dev_hor_vector,  host_vettore_hor,  totalbytes, cudaMemcpyHostToDevice );
	cudaMemcpy( dev_hortype_vector, host_vettore_hortype, totalbytes, cudaMemcpyHostToDevice );
	CudaCheckError();

	
	// Liberiamo memoria che non sarà più utilizzata
	free( left_matrix );
	free( right_matrix );
	free( var_matrix );
	// free( rav_matrix );
	free( fear_matrix );
	free( host_vettore_hor );
	free( host_vettore_hortype );

	return;
}

void read_feeds( std::string folder, unsigned int species, unsigned int parallel_threads , unsigned int pblocks, unsigned int off_begin, unsigned int* totale_specie_feed, bool verbose ) {

	/* 
		La matrice dei feed funziona così: 
		le specie in feed sono sempre le stesse, cambiano le sole quantità
	*/

	std::string qty_path = folder+"/MX_feed";
	if (verbose) std::cout << " * Reading general feed quantities from " << qty_path << "\n";
	std::string line, token;
	std::vector<std::string> vet1; 
	std::ifstream in(qty_path.c_str());
	std::stringstream ss;
	
	if (!in.is_open()) {
		// perror("ERROR: cannot read feed quantities");
		printf("WARNING: cannot read feed quantities\n");
		return;
	}


	// pre-processing: identify feed species
	vet1.clear();
	while(getline(in,line)) { 
		vet1.push_back(line);
	}

	ss.clear();
	ss << vet1.back();		// offset
	*totale_specie_feed=0;
	unsigned int s=0;
	std::vector<unsigned int> specie_in_feed;
	while ( std::getline(ss, token, '\t') ) {
		unsigned qty = atoi(token.c_str());
		if (qty!=0) {
			// printf("FEED identified: species %d, value %d\n",s,qty);
			(*totale_specie_feed)++;
			specie_in_feed.push_back(s);
		}
		s++;
	}
	*totale_specie_feed = specie_in_feed.size();

	if ( *totale_specie_feed == 0 ) {
		if (verbose) printf(" * No species have feed\n");
		return;
	}

	size_t totalbytes = sizeof(unsigned int) * *totale_specie_feed * parallel_threads * pblocks;
	unsigned int* feed_matrix =(unsigned int*) malloc( totalbytes );
	
	// droppo le righe fuori intervallo
	for ( unsigned int i=0; i<pblocks; i++) {
		// printf ("Loading block %d\n", i);
		for ( unsigned int t=0; t<parallel_threads; t++ ) {
			// printf (" Loading thread %d, position %d\n", t, i*parallel_threads+t+off_begin);
			ss.clear();
			ss << vet1[i*parallel_threads+t+off_begin];		// offset
			unsigned int s=0;
			unsigned int pos_m =0;
			// printf ("  Loading species %d, %d, %d\n", i,t,s);
			while ( std::getline(ss, token, '\t') ) {

				if (pos_m<specie_in_feed.size()) {						

					if (s == specie_in_feed[pos_m]) {					
						unsigned int res;
						std::stringstream convert(token);
						convert >> res;				
						// qty_matrix[ACCESS_SPECIES] = (unsigned int)res;
						feed_matrix[ACCESS_FEED_SPECIES] = res;
						pos_m++;
					}

				}
								
				s++;
			}
			if (s!=species) {
				printf("ERROR! too many quantities %d (riga %d)\n", s, i);
				exit(-1);
			}
				
			// printf("\n");
		}
	} // end for

	cudaMemcpyToSymbol( DEV_CONST_FEEDSPECIES,   totale_specie_feed, sizeof(totale_specie_feed));
	CudaCheckError();

	unsigned int f_back = 0;
	cudaMemcpyFromSymbol( &f_back, DEV_CONST_FEEDSPECIES, sizeof(f_back) );
	if (verbose) printf( " * Verified on GPU: number of species in feed was set to %d\n", f_back);


	cudaMalloc( &dev_feed_indeces, sizeof(unsigned int) * *totale_specie_feed );
	CudaCheckError();
	cudaMalloc( &dev_feed_values,  totalbytes );
	CudaCheckError();

	cudaMemcpy( dev_feed_indeces, &specie_in_feed[0],  sizeof(unsigned int) * *totale_specie_feed , cudaMemcpyHostToDevice);
	CudaCheckError();
	cudaMemcpy( dev_feed_values, feed_matrix, totalbytes, cudaMemcpyHostToDevice);
	CudaCheckError();

	specie_in_feed.clear();	
	free(feed_matrix);

}


void read_quantities_offset( std::string folder, unsigned int reactions, unsigned int species, unsigned int parallel_threads, unsigned int pblocks, unsigned int off_begin, unsigned int off_end, bool verbose ) {

	std::string qty_path = folder+"/MX_0";
	if (verbose) std::cout << " * Reading initial quantities for all the blocks from " << qty_path << " with offset " << off_begin << "\n";
	std::string line, token;
	std::vector<std::string> vet1, vet2;
	std::ifstream in(qty_path.c_str());
	std::stringstream ss;

	unsigned int* qty_matrix;
	unsigned int totalbytes;
	// unsigned int parallel_threads = off_end - off_begin;

	// pre-processing
	if ( in.is_open() ) {

		// brief preprocessing to determine the dimensions
		vet1.clear();
		while(getline(in,line)) {
			if (line.compare("\n")==0) {
			//	printf("Linea vuota detected\n");
				continue;
			}
			vet1.push_back(line);
		}

		if(  (off_end - off_begin) > vet1.size() ) {
			printf("ERROR: input files rows (%d) and requested threads (%d) do not fit\n", vet1.size(), (off_end - off_begin));
			exit(-4);
		}

		// printf(" * Found %d different initial conditions...\n", vet1.size() );
				
		totalbytes = sizeof(unsigned int) * species * (off_end - off_begin);
		qty_matrix = (unsigned int*) malloc ( totalbytes );
		memset(qty_matrix, 0, totalbytes);
		if (verbose) printf(" * Will use just %d initial conditions\n", off_end - off_begin);

		// droppo le righe fuori intervallo
		for ( unsigned int i=0; i<pblocks; i++) {
			// printf ("Loading block %d\n", i);
			for ( unsigned int t=0; t<parallel_threads; t++ ) {
				// printf (" Loading thread %d, position %d\n", t, i*parallel_threads+t+off_begin);
				ss.clear();
				ss << vet1[i*parallel_threads+t+off_begin];		// offset
				unsigned int s=0;
				// printf ("  Loading species ", i);
				while ( std::getline(ss, token, '\t') ) {									
					// printf ("%d\t", s);
					// convert token in char				
					double res;
					std::stringstream convert(token);
					convert >> res;				
					// qty_matrix[ACCESS_SPECIES] = (unsigned int)res;
					qty_matrix[ACCESS_SPECIES] = (unsigned int)res;
					// printf("%d\n", qty_matrix[ACCESS_SPECIES] );
					s++;
				}
				if (s!=species) {
					printf("ERROR! too many quantities %d (riga %d)\n", s, i);
					exit(-5);
				}		
				
				// printf("\n");
			}
		} // end for

		cudaMalloc( &dev_qty_matrix,  totalbytes );
		if (verbose) printf(" * Allocated %d bytes for quantities\n", totalbytes);
		CudaCheckError();
		cudaMemcpy( dev_qty_matrix, qty_matrix,  totalbytes, cudaMemcpyHostToDevice );
		CudaCheckError();
		
	} else {
		perror("ERROR reading quantities matrix");
		// system("pause");
		exit(-6);
	}

}

void read_quantities( std::string folder, unsigned int reactions, unsigned int species, unsigned int parallel_threads, unsigned int* parallel_blocks, bool verbose ) {

	std::string qty_path = folder+"/X_0";
	if (verbose) std::cout << " * Reading initial quantities for all the blocks from " << qty_path << "\n";
	std::string s, line, token;
	std::vector<std::string> vet1, vet2;
	std::ifstream in(qty_path.c_str());
	std::stringstream ss;

	unsigned int* qty_matrix;
	unsigned int totalbytes;

	// pre-processing
	if ( in.is_open() ) {

		// brief preprocessing to determine the dimensions
		vet1.clear();
		while(getline(in,line)) { 
			vet1.push_back(line);
		}

		unsigned int pblocks = vet1.size();
		if (verbose) printf(" * Found %d different initial conditions, %d threads each, loading... ", pblocks, parallel_threads );
		*parallel_blocks = pblocks;

		totalbytes = sizeof(unsigned int)* species * parallel_threads * pblocks;

		qty_matrix = (unsigned int*) malloc ( totalbytes );
		memset(qty_matrix, 0, totalbytes);

		for ( unsigned int i=0; i<pblocks; i++ ) {
			ss.clear();
			ss << vet1[i];
			unsigned int s=0;
			while ( std::getline(ss, token, '\t') ) {
				// convert token in char				
				double res;
				std::stringstream convert(token);
				convert >> res;				
				for ( unsigned int t=0; t<parallel_threads; t++ ) {
					qty_matrix[ACCESS_SPECIES] = (unsigned int)res;
				}				
				s++;
			}
			if (s>species) {
				printf("ERROR! too many quantities\n");
				exit(-1);
			}
		}
		if (verbose) printf(" quantities loaded!\n");

	} else {
		perror("ERROR reading quantities matrix");
		// system("pause");
		exit(-1);
	}

	// finally, we load everything on the GPU
	cudaMalloc( &dev_qty_matrix,  totalbytes );
	CudaCheckError();

	cudaMemcpy( dev_qty_matrix, qty_matrix,  totalbytes, cudaMemcpyHostToDevice );
	CudaCheckError();
		
}

void read_parameters_offset( std::string folder, unsigned int reactions, unsigned int parallel_threads, unsigned int pblocks, unsigned int off_begin, unsigned int off_end, bool verbose ) {

	std::string par_path = folder+"/c_matrix";
	if (verbose) std::cout << " * Reading parameters for all the blocks from " << par_path << "\n";
	std::string s, line, token;
	std::vector<std::string> vet1, vet2;
	std::ifstream in(par_path.c_str());
	std::stringstream ss;

	param_t* par_matrix;
	unsigned int totalbytes;

	// pre-processing
	if ( in.is_open() ) {

		// brief preprocessing to determine the dimensions
		vet1.clear();
		while(getline(in,line)) { 
			vet1.push_back(line);
		}
			
		if (verbose) printf(" * Loading parameters, " );
		
		totalbytes = sizeof(param_t) * reactions * (off_end - off_begin);
		par_matrix = (param_t*) malloc ( totalbytes );
		memset(par_matrix, 0, totalbytes);
		if (verbose) printf("will use %d parameters\n", off_end - off_begin);
		
		// droppo le righe fuori intervallo
		for ( unsigned int i=0; i<pblocks; i++) {
			for ( unsigned int t=0; t<parallel_threads; t++ ) {
				ss.clear();
				ss << vet1[i*parallel_threads+t+off_begin];		// offset
				unsigned int r=0;
				while ( std::getline(ss, token, '\t') ) {
					// convert token in char				
					// std::cout << token << "\t";
					// if (token.size()<2) continue;
					double res;
					std::stringstream convert(token);
					convert >> res;				
					par_matrix[ACCESS_REACTIONS] = res;
					r++;
				}
				if (r!=reactions) {
					printf("ERROR! too many parameters (%d)\n", r);
					exit(-1);
				}		
			}
		}
		// printf("done!\n");

	} else {
		perror("ERROR reading quantities matrix");
		// system("pause");
		exit(-1);
	}

	// finally, we load everything on the GPU
	cudaMalloc( &dev_par_matrix,  totalbytes );
	CudaCheckError();

	cudaMemcpy( dev_par_matrix, par_matrix,  totalbytes, cudaMemcpyHostToDevice );
	CudaCheckError();

}

void read_common_parameters( std::string folder, unsigned int reactions) {

	std::string par_path = folder+"/c_vector";
	std::cout << " * Reading parameters for all the blocks from " << par_path << "\n";
	std::string s, line, token;
	std::vector<std::string> vet1, vet2;
	std::ifstream in(par_path.c_str());
	std::stringstream ss;

	param_t* par_matrix;

	// pre-processing
	if ( in.is_open() ) {

		// brief preprocessing to determine the dimensions
		vet1.clear();
		while(getline(in,line)) { 
			vet1.push_back(line);
		}
		if (vet1.size() != reactions) {
			printf("ERROR: number of parameters is incompatible with reactions\n");
			exit(-1);
		}
		
		printf(" * Loading parameters... \n" );

		par_matrix = (param_t*) malloc ( sizeof(param_t) * reactions  );  

		param_t q;
		for ( unsigned int i=0; i<vet1.size(); i++ ) {
			std::stringstream convert(vet1[i]);
			convert >> q;
			par_matrix[ i ] = q; 
		}

		//printf("done!\n");

	} else {
		perror("ERROR reading quantities matrix");
		// system("pause");
		exit(-1);
	}

	// finally, we load everything on the GPUs
	size_t totalbytes = sizeof(param_t)* reactions ;

	cudaMalloc( &dev_par_matrix,  totalbytes );
	CudaCheckError();

	cudaMemcpy( dev_par_matrix, par_matrix,  totalbytes, cudaMemcpyHostToDevice );
	CudaCheckError();

}

void read_time_max( std::string folder, tau_t* tempo, bool verbose ) {

	std::string time_path = folder+"/time_max";
	if (verbose) std::cout << " * Reading time max from " << time_path << "\n";
	std::string s, line, token;
	std::vector<std::string> vet1, vet2;
	std::ifstream in(time_path.c_str());
	std::stringstream ss;

	// pre-processing
	if ( in.is_open() ) {

		getline(in,line);
		tau_t q;
		std::stringstream convert(line);
		convert >> q;
		*tempo = q;
		if (verbose) printf( " * Time max = %f\n", q);

	} else {
		perror("ERROR reading quantities matrix");
		// system("pause");
		exit(-1);
	}

}

void set_symbols( unsigned int s, unsigned int r,  tau_t t, unsigned int samples, unsigned int samplespecies,
				 unsigned int  repetitions, unsigned int experiments, bool verbose ) {
					 
	cudaMemcpyToSymbol( DEV_CONST_EXPERIMENTS , &experiments, sizeof(experiments));
	CudaCheckError();
	unsigned int e_back = 0;
	cudaMemcpyFromSymbol( &e_back, DEV_CONST_EXPERIMENTS, sizeof(e_back) );
	CudaCheckError();
	if (verbose) printf( " * Verified on GPU: DEV_CONST_EXPERIMENTS was set to %d\n", e_back);
	

	cudaMemcpyToSymbol( DEV_CONST_SPECIES,   &s, sizeof(s));
	CudaCheckError();
	unsigned int s_back = 0;
	cudaMemcpyFromSymbol( &s_back, DEV_CONST_SPECIES, sizeof(s_back) );
	CudaCheckError();
	if (verbose) printf( " * Verified on GPU: species was set to %d\n", s_back);
	
	
	cudaMemcpyToSymbol( DEV_CONST_REACTIONS, &r, sizeof(r));
	CudaCheckError();
	unsigned int r_back = 0;
	cudaMemcpyFromSymbol( &r_back, DEV_CONST_REACTIONS, sizeof(r_back) );
	CudaCheckError();
	if (verbose) printf( " * Verified on GPU: reactions was set to %d\n", r_back);
	

	cudaMemcpyToSymbol( DEV_CONST_TIMEMAX,   &t, sizeof(t));
	CudaCheckError();
	tau_t t_back = 0;
	cudaMemcpyFromSymbol( &t_back, DEV_CONST_TIMEMAX, sizeof(t_back) );
	CudaCheckError();
	if (verbose) printf( " * Verified on GPU: time_max was set to %f\n", t_back);
	
	cudaMemcpyToSymbol( DEV_CONST_SAMPLES,  &samples, sizeof(samples));
	CudaCheckError();
	unsigned int samples_back = 0;
	cudaMemcpyFromSymbol( &samples_back, DEV_CONST_SAMPLES, sizeof(samples_back) );
	CudaCheckError();
	if (verbose) printf( " * Verified on GPU: number of samples was set to %d\n", samples_back);

	cudaMemcpyToSymbol( DEV_CONST_SAMPLESPECIES,  &samplespecies, sizeof(samplespecies));
	CudaCheckError();
	unsigned int ss_back = 0;
	cudaMemcpyFromSymbol( &ss_back, DEV_CONST_SAMPLESPECIES, sizeof(ss_back) );
	CudaCheckError();
	if (verbose) printf( " * Verified on GPU: number of samples species was set to %d\n", ss_back);

	/*
	cudaMemcpyToSymbol( DEV_CONST_SAMPLESLUN , &samplespecies, sizeof(samplespecies));
	CudaCheckError();
	unsigned int sspecies_back = 0;
	cudaMemcpyFromSymbol( &sspecies_back, DEV_CONST_SAMPLESLUN, sizeof(sspecies_back) );
	CudaCheckError();
	if (verbose) printf( " * Verified on GPU: DEV_CONST_SAMPLESLUN was set to %d\n", sspecies_back);
	*/

	cudaMemcpyToSymbol( DEV_CONST_REPETITIONS , &repetitions, sizeof(unsigned int ));
	CudaCheckError();
	unsigned int rep_back = 0;	
	cudaMemcpyFromSymbol( &rep_back, DEV_CONST_REPETITIONS, sizeof(rep_back) );
	CudaCheckError();
	if (verbose) printf( " * Verified on GPU: DEV_CONST_REPETITIONS  was set to %d\n", rep_back);


}

void read_instants( std::string MODEL_FOLDER, unsigned int *SAMPLES, unsigned int *SAMPLESPECIES, unsigned int species, tau_t *timemax, bool verbose ) {

	std::string par_path = MODEL_FOLDER+"/t_vector";
	if (verbose) std::cout << " * Reading sampling instants from " << par_path << "\n";
	std::string s, line, token;
	std::vector<std::string> vet1, vet2;
	std::ifstream in(par_path.c_str());
	std::stringstream ss;

	std::vector<tau_t> samplesv;

	// pre-processing
	if ( in.is_open() ) {

		// brief preprocessing to determine the dimensions
		vet1.clear();
		while(getline(in,line)) { 
			vet1.push_back(line);
		}
		
		param_t q;
		for ( unsigned int i=0; i<vet1.size(); i++ ) {
			std::stringstream convert(vet1[i]);
			convert >> q;
			// par_matrix[ i ] = q; 
			samplesv.push_back(q);
						
		}

	} else {
		perror("ERROR reading samples vector");
		// system("pause");
		exit(-1);
	}

	in.close();

	*timemax = samplesv.back();

	if (verbose) printf(" * %d temporal instants required, up to t=%f.\n", samplesv.size(), timemax);

	// finally, we load everything on the GPU
	*SAMPLES = samplesv.size();
	size_t totalbytes = sizeof(tau_t)* *SAMPLES;
	
	


	cudaMemcpyToSymbol( DEV_CONST_TIMESLUN, SAMPLES, sizeof(SAMPLES));
	CudaCheckError();


	cudaMalloc( &dev_time_vector,  totalbytes );
	CudaCheckError();

	cudaMemcpy( dev_time_vector, &samplesv[0],  totalbytes, cudaMemcpyHostToDevice );
	CudaCheckError();

	host_sampletime = (tau_t*) malloc (totalbytes);
	memcpy(host_sampletime, &samplesv[0], totalbytes);


	/// NEW!!
	

	std::string cs_path = MODEL_FOLDER+"/cs_vector";
	if (verbose) std::cout << " * Reading chemical species to sample from " << cs_path  << "\n";
	in.open(cs_path.c_str());

	std::vector<unsigned int> samplescs;

	if (verbose) printf(" * Reading chemical species to sample:\n");
	// pre-processing
	if ( in.is_open() ) {

		// brief preprocessing to determine the dimensions
		vet1.clear();
		while(getline(in,line)) { 
			vet1.push_back(line);
		}
		
		unsigned int q;
		for ( unsigned int i=0; i<vet1.size(); i++ ) {
			std::stringstream convert(vet1[i]);
			convert >> q;
			// par_matrix[ i ] = q; 
			samplescs.push_back(q);
			if (verbose) printf("\t%d", q);
		}

	} else {
		// perror("ERROR reading chemical species to sample vector");
		printf ("WARNING: chemical species to sample vector not found, saving ALL species as default.\n");
		// system("pause");
		// exit(-1);
		for (unsigned int i=0; i<species; i++) {
			samplescs.push_back(i);
		}
	}

	if (verbose) printf("\n * %d species will be sampled\n", samplescs.size());
	*SAMPLESPECIES = samplescs.size();

	


	// finally, we load everything on the GPU	
	totalbytes = sizeof(unsigned int)* (samplescs.size());
	cudaMalloc( &dev_species_to_sample_vector,  totalbytes );
	CudaCheckError();

	cudaMemcpy( dev_species_to_sample_vector, &samplescs[0],  totalbytes, cudaMemcpyHostToDevice );
	CudaCheckError();

	host_species_to_sample_vector = (unsigned int*) malloc (totalbytes);
	memcpy(host_species_to_sample_vector, &samplescs[0], totalbytes);

	in.close();

	// delete( samplesv );

}


void read_time_series(std::string MODEL_FOLDER,  unsigned int THREADS, unsigned int BLOCKS, unsigned int * REPETITIONS, unsigned int* PARREPETITIONS, 
	unsigned int* TARGET_QUANTITIES, unsigned int* EXPERIMENTS, unsigned int SAMPLES, unsigned int SAMPLESPECIES, 
	bool dump=false) {

	unsigned int threads = THREADS * BLOCKS;
	
	std::string v;	

	// ogni vettore è associato alla sua condizione sperimentale (aka sciame)
	thread2experiment = (char*) malloc ( sizeof(char)*threads );


	/// Step ??: apertura serie temporali (if any)
	/// la serie è lunga T_vector
	/// dobbiamo leggere le informazioni su il numero di esperimenti, e ripetizioni
	std::ifstream ts_rep_file;
	ts_rep_file.open( (MODEL_FOLDER+"/ts_rep").c_str() );
	if (! ts_rep_file.is_open() ) {
		perror("WARNING: cannot open ts_rep file: ");
	} else {
		getline(ts_rep_file, v);
		*REPETITIONS = atoi(v.c_str());	
		if (dump) printf(" * Repetitions is %d\n", *REPETITIONS);
	
		ts_rep_file.close();
	}


	/// Step ??: numero di ripetizioni parallele (G), valore singolo
	std::ifstream ts_parrep_file;
	ts_parrep_file.open( (MODEL_FOLDER+"/p_rep").c_str() );
	if (! ts_parrep_file.is_open() ) {
		perror("WARNING: cannot open p_rep file: ");
	} else {
		getline(ts_parrep_file, v);
		*PARREPETITIONS = atoi(v.c_str());	

		cudaMemcpyToSymbol( DEV_CONST_PAR_REPETITIONS , PARREPETITIONS, sizeof(PARREPETITIONS));
		CudaCheckError();

		ts_parrep_file.close();
	}


	std::ifstream ts_numtgt_file;
	ts_numtgt_file.open( (MODEL_FOLDER+"/ts_numtgt").c_str() );
	if (! ts_numtgt_file.is_open() ) {
		perror("WARNING: cannot open ts_numtgt file: ");
	} else {
		getline(ts_numtgt_file, v);
		*TARGET_QUANTITIES = atoi(v.c_str());	

		cudaMemcpyToSymbol( DEV_CONST_TARGET_QUANTITIES , TARGET_QUANTITIES, sizeof(TARGET_QUANTITIES));
		CudaCheckError();

		ts_numtgt_file.close();
	}

	
	
	std::ifstream tts_file;
	tts_file.open( (MODEL_FOLDER+"/tts_vector").c_str() );
	if (! tts_file.is_open() ) {
		perror("WARNING: cannot open tts_vector file: ");
	} else {
		char pp=0;
		unsigned int last=0;
		while( tts_file.good() ) {
			getline(tts_file, v);
			if (v.length()<1) continue;
			for (unsigned int kk=last; kk<atoi(v.c_str()); kk++) {
				thread2experiment[kk]=pp;
			}
			last =  atoi(v.c_str());
			pp++;
			// this->experiments = atoi(v.c_str());	
		}		
		
		*EXPERIMENTS = pp;

		if (last!=THREADS*BLOCKS) {
			printf("ERROR! cannot assign threads (%d) to initial conditions (%d), aborting\n", THREADS*BLOCKS, last);
			system("pause");
			exit(-10);
		}

		/*
		if (dump) {
			printf(" * Threads assigned to conditions:\n");
			for (unsigned kk=0; kk<threads; kk++) {
				printf("%d\t", thread2experiment[kk]);
			}
			printf("\n");
		}
		*/
		tts_file.close();
	}

	


	if (dump) printf(" * Experiments: %d, repetitions: %d target quantities: %d\n", *EXPERIMENTS, *REPETITIONS, *TARGET_QUANTITIES);


	/// sappiamo le righe (len time_instants), sappiamo le colonne (exp * rep * len cs_vector) 
	global_time_series = (unsigned int*) malloc ( sizeof(unsigned int) * 
		*EXPERIMENTS * 
			*REPETITIONS * 
				*TARGET_QUANTITIES * 
					SAMPLES );


	std::ifstream ts_matrix_file;
	std::vector<std::string> a;
	ts_matrix_file.open( (MODEL_FOLDER+"/ts_matrix").c_str() );
	unsigned int riga =0;
	if (! ts_matrix_file.is_open() ) {
		perror("WARNING: cannot open ts file (fitness unavailable)");
	} else {
		
		
		unsigned int colonne = *EXPERIMENTS * *REPETITIONS * *TARGET_QUANTITIES;

		while( ts_matrix_file.good() ) {

			getline(ts_matrix_file, v);
			if (v.length()<2) continue;
			a.clear();
			tokenize(v,a);
			for (unsigned int kk=1; kk<colonne+1; kk++ ) {
				global_time_series[ riga*colonne + kk-1 ] = atoi(a[kk].c_str());
			}
			riga++;

		}

		if (dump) printf(" * Time series loaded and assigned to threads\n");
		ts_matrix_file.close();

	}

	// the "+1" is the cumulative normalizer
	host_normalizers = (float*) malloc (sizeof(float)* *EXPERIMENTS * (*TARGET_QUANTITIES+1) );
	memset( host_normalizers, 0, sizeof(float)* *EXPERIMENTS * (*TARGET_QUANTITIES+1) );
	
	/// calculate normalizers
	for (unsigned int d=0; d<*EXPERIMENTS; d++) {
		for (unsigned int s=0; s<*TARGET_QUANTITIES; s++) {
			for (unsigned int t=0; t<SAMPLES; t++) {
				float accum = 0;
				for (unsigned int e=0; e<*REPETITIONS; e++) {		
					unsigned int vl= global_time_series[  
						t*( (*TARGET_QUANTITIES)* *REPETITIONS * *EXPERIMENTS ) + 
						d*( (*TARGET_QUANTITIES)* *REPETITIONS) + 	
						e*( (*TARGET_QUANTITIES)  ) + s ]; 
					accum += vl;
				}
				accum /= *REPETITIONS;
				host_normalizers[d* (*TARGET_QUANTITIES+1) + s ] += accum;
			}
			host_normalizers[d* (*TARGET_QUANTITIES+1) + s ] /= SAMPLES;
			host_normalizers[d* (*TARGET_QUANTITIES+1) + *TARGET_QUANTITIES] += host_normalizers[d* (*TARGET_QUANTITIES+1) + s ];
		}
	}
	
	for (unsigned int d=0; d<*EXPERIMENTS; d++) {
		if (dump) printf(" * Cumulative integral swarm %d: %f.\n", d, host_normalizers[d* (*TARGET_QUANTITIES+1) + *TARGET_QUANTITIES] );
	}

	for (unsigned int d=0; d<*EXPERIMENTS; d++) {
		for (unsigned int s=0; s<*TARGET_QUANTITIES; s++) {
			host_normalizers[d* (*TARGET_QUANTITIES+1) + s ] /= host_normalizers[d* (*TARGET_QUANTITIES+1) + *TARGET_QUANTITIES];
		}
	}

	// dump normalizers
	for (unsigned int d=0; d<*EXPERIMENTS; d++) {
		for (unsigned int s=0; s<*TARGET_QUANTITIES; s++) {
			if (dump) printf("Alpha_{d%d,s%d}=%f\t", d, s, host_normalizers[d* (*TARGET_QUANTITIES+1) + s ] );
		}
		printf("\n");
	}

	cudaMalloc( &dev_normalizers, sizeof(float)* *EXPERIMENTS * (*TARGET_QUANTITIES+1));
	CudaCheckError();
	cudaMemcpy( dev_normalizers, host_normalizers, sizeof(float)* *EXPERIMENTS * (*TARGET_QUANTITIES+1), cudaMemcpyHostToDevice);
	CudaCheckError();


	if (SAMPLES != riga ) {

		printf("ERROR! t_vector (%d) and ts_matrix (%d) have different size\n", SAMPLES, riga);
		// exit(-12);

	}

	// alloco

	cudaMalloc((void**)&device_target, sizeof( unsigned int ) * *REPETITIONS * *EXPERIMENTS * *TARGET_QUANTITIES * SAMPLES );
	CudaCheckError();
	cudaMemcpy(device_target, global_time_series, sizeof( unsigned int ) * *REPETITIONS * *EXPERIMENTS * *TARGET_QUANTITIES * SAMPLES ,cudaMemcpyHostToDevice);
	CudaCheckError();



}



/************************************************************************************************************************************************************/
/************************************************************************************************************************************************************/


__global__ void PrepareVariables( unsigned int* SQ, unsigned int* X, tau_t* T, unsigned int* checksample, unsigned int * samples, char* SSA, bool FORCE_SSA ) {

	// initialize threads	
	unsigned int gid = threadIdx.x + blockIdx.x*blockDim.x;
	unsigned int larg = blockDim.x * gridDim.x;

	// init time
	T[gid] = 0;
	checksample[gid] = 0;

	for (unsigned int i=0; i<DEV_CONST_SAMPLES; i++) 
		for (unsigned int s=0; s<DEV_CONST_SAMPLESPECIES; s++)
			samples[larg*DEV_CONST_SAMPLESPECIES*i + larg*s + gid] = 0;

	// init species amounts coalesced	
	for (unsigned int s=0; s<DEV_CONST_SPECIES; s++) X[ GLOBAL_S ] = SQ[ GLOBAL_S ]; 
	
	/*
#ifdef DO_TAU_STEPS 
	SSA[gid] = 0;
#else
	SSA[gid] = 1;
#endif
	*/

	if (FORCE_SSA) SSA[gid]=1;
	else           SSA[gid]=0;

}

// template <int action>
__device__ bool test_update(  stoich_t* var) {

	extern __shared__ sh_type global_array[];

	int tid = threadIdx.x;
	int bdim = blockDim.x;

	unsigned int*	X		= (unsigned int*)global_array;	
	param_t*		PAR		= (param_t*)&X[DEV_CONST_SPECIES*bdim];				
	param_t*		PROP	= (param_t*)&PAR[DEV_CONST_REACTIONS*bdim];	
	unsigned int*   K_rule	= (unsigned int*)&PROP[DEV_CONST_REACTIONS*bdim];
	// g_t*			G		= (g_t*)&K_rule[DEV_CONST_REACTIONS*bdim];
	int*			X_1		= (int*)&K_rule[DEV_CONST_REACTIONS*bdim];	
	char*			CRIT	= (char*)&X_1[DEV_CONST_SPECIES*bdim];
	// tau_t*			S_INST  = (tau_t*)&CRIT[DEV_CONST_REACTIONS*bdim];

	for ( unsigned int s=0; s<DEV_CONST_SPECIES; s++ )  {
		X_1[ SHARED_S ] = X[ SHARED_S ]; 
	}
	
#ifdef USE_CONSTANT_FEAR

	for (unsigned int h=0; h<DEV_CONST_FLATTEN_FEAR; h++) {

		char r= DEV_CONST_MAT_FEARFLAT[h].x;
		char s= DEV_CONST_MAT_FEARFLAT[h].y;		

		X_1[ SHARED_S ] += DEV_CONST_MAT_FEARFLAT[h].z * K_rule[SHARED_R] ;		
		
		if ( X_1[ SHARED_S ]<0 )	return false;

	}


	#else
	for ( unsigned int r=0; r<DEV_CONST_REACTIONS; r++ ) {
		for ( unsigned int s=0; s<DEV_CONST_SPECIES; s++ ) {			
			// X_1[ SHARED_S ] += ( var[ r*DEV_CONST_SPECIES + s ] * K_rule[SHARED_R] );
			// X_1[ SHARED_S ] += ( DEV_CONST_MAT_FEAR[ r*DEV_CONST_SPECIES + s ] * K_rule[SHARED_R] );	

			X_1[ SHARED_S ] += ( var[ r*DEV_CONST_SPECIES + s ] * K_rule[SHARED_R] );	

			
			//X_1[ SHARED_S ] = 1;
			if ( X_1[ SHARED_S ]<0 )		return false;
		}
	}
#endif 

	return true;

}

template<int action>
__device__ param_t CalculatePropensity(  unsigned int* X, param_t* new_prob_a, const param_t* parametri, const stoich_t* left_side, const stoich_f* left_flatten ) {

	int tid = threadIdx.x;
	unsigned int bdim = blockDim.x;
	unsigned int larg = bdim * gridDim.x;
	unsigned int gid  = tid + bdim*blockIdx.x;
	// int gid = threadIdx.x + blockDim.x * blockIdx.x;

	// azzero a0
	param_t a0 = 0.0f;
	
	switch (action) {

		case DEV_NORMAL_MATRIX:

			// per ogni reazione..
			for (unsigned r=0; r<DEV_CONST_REACTIONS; r++) {

				/// i parametri sono le posizioni delle particelle. 
				/// sono gli stessi per tutte le ripetizioni della simulazione.
				new_prob_a[ SHARED_R ] = parametri[ SHARED_R ];
		
				// per ogni specie..
				for (unsigned int s=0; s<DEV_CONST_SPECIES; s++) {
			
					stoich_t val_sx = left_side[ r*DEV_CONST_SPECIES + s ];

					// forse ottimizza, forse no.
					if ( val_sx > 0 ) {

						if ( val_sx == 1 ) {

							new_prob_a[ SHARED_R ] *= X[ SHARED_S ];	

						} else if ( val_sx == 2 ) {

							new_prob_a[ SHARED_R ] *=  X[ SHARED_S ];
							new_prob_a[ SHARED_R ] *= (X[ SHARED_S ]-1);
							new_prob_a[ SHARED_R ] *= 0.5f;
					
						} else {
					
							unsigned int e;
							for ( e=1; e<=val_sx; e++ )
								new_prob_a[ SHARED_R ] *= ( (param_t) ( X[ SHARED_S ] - e + 1) ) / ( (param_t) e );

						} 

					} 

				} // end for c


				a0 += new_prob_a[ SHARED_R ];
				// printf("GPU prop> %f\n", new_prob_a[ SHARED_R ]);

			} // end for r

		break;

		case DEV_FLATTEN_MATRIX:

			for (unsigned int r=0; r<DEV_CONST_REACTIONS; r++) 
				new_prob_a[ SHARED_R ] = parametri[ SHARED_R ];
					

			for (unsigned int f=0; f<DEV_CONST_FLATTEN_LEFT; f++) {		
				
				stoich_f u = left_flatten[f];
				unsigned int r = u.x;
				unsigned int s = u.y;
				int v = u.z; 
				
				if (v==1) new_prob_a[ SHARED_R ] *= X[SHARED_S];  
				else if  ( v==2 ) {
					new_prob_a[ SHARED_R ] *= ( 0.5f * X[ SHARED_S ] * (X[ SHARED_S ]-1) );
				} else {
					unsigned int e;
						for ( e=1; e<=u.z; e++ )
							new_prob_a[ SHARED_R ] *= ( (param_t) ( X[ SHARED_S ] - e + 1) ) / ( (param_t) e );
				}		
								
			} 

			for (unsigned int r=0; r<DEV_CONST_REACTIONS; r++)
				a0 += new_prob_a[ SHARED_R ];

		break;

		case DEV_FLATTEN_MATRIX_CONST:

			for (unsigned int r=0; r<DEV_CONST_REACTIONS; r++) 
				new_prob_a[ SHARED_R ] = parametri[ SHARED_R ];
					

			for (unsigned int f=0; f<DEV_CONST_FLATTEN_LEFT; f++) {		
				
				stoich_f u = DEV_CONST_MAT_LEFTFLAT[f];
				unsigned int r = u.x;
				unsigned int s = u.y;
				int v = u.z; 
				
				if (v==1) new_prob_a[ SHARED_R ] *= X[SHARED_S];  
				else if  ( v==2 ) {
					new_prob_a[ SHARED_R ] *= ( 0.5f * X[ SHARED_S ] * (X[ SHARED_S ]-1) );
				} else {
					unsigned int e;
						for ( e=1; e<=u.z; e++ )
							new_prob_a[ SHARED_R ] *= ( (param_t) ( X[ SHARED_S ] - e + 1) ) / ( (param_t) e );
				}		
								
			} 

			for (unsigned int r=0; r<DEV_CONST_REACTIONS; r++)
				a0 += new_prob_a[ SHARED_R ];

		break;

		case DEV_FLATTEN_MATRIX_CONST_GLOBPARAMS:

			for (unsigned int r=0; r<DEV_CONST_REACTIONS; r++) 
				new_prob_a[ SHARED_R ] = parametri[ GLOBAL_R ];
					

			for (unsigned int f=0; f<DEV_CONST_FLATTEN_LEFT; f++) {		
				
				stoich_f u = DEV_CONST_MAT_LEFTFLAT[f];
				unsigned int r = u.x;
				unsigned int s = u.y;
				int v = u.z; 
				
				if (v==1) new_prob_a[ SHARED_R ] *= X[SHARED_S];  
				else if  ( v==2 ) {
					new_prob_a[ SHARED_R ] *= ( 0.5f * X[ SHARED_S ] * (X[ SHARED_S ]-1) );
				} else {
					unsigned int e;
						for ( e=1; e<=u.z; e++ )
							new_prob_a[ SHARED_R ] *= ( (param_t) ( X[ SHARED_S ] - e + 1) ) / ( (param_t) e );
				}		
								
			} 

			for (unsigned int r=0; r<DEV_CONST_REACTIONS; r++)
				a0 += new_prob_a[ SHARED_R ];

		break;

	}

	return a0;

}

__device__ void check_time_instant( tau_t tempoprima, tau_t tau, tau_t* samplestimes, 
									unsigned int* X, int* X1, 
									unsigned int* checkedsample, unsigned int * storage, unsigned int* species_to_sample_vector 
									) {

	unsigned int tid  = threadIdx.x;
	unsigned int bdim = blockDim.x;
	unsigned int gid  = tid + bdim*blockIdx.x;
	unsigned int larg = bdim * gridDim.x;

	unsigned int i = checkedsample[gid];
	if ( i<DEV_CONST_SAMPLES ) {

		// per tutti gli istanti coperti dal salto temporale
		while( tempoprima + tau >= samplestimes[i] ) {
			
			// printf(" * Reading istante %d/%d (%f u.a.)\n", i, DEV_CONST_SAMPLES,  samplestimes[i]);

			tau_t local_st = samplestimes[i];
			// unsigned int larg = bdim * gridDim.x;
			unsigned int partial_offset = larg*DEV_CONST_SAMPLESPECIES*i + gid; // larg*s 

			// segno l'interpolazione di tutte le specie chimiche
			// for (unsigned int s=0; s<DEV_CONST_SPECIES; s++) {			
			for (unsigned int sk=0; sk<DEV_CONST_SAMPLESPECIES; sk++) {			
				unsigned int s = species_to_sample_vector[sk];
				
				/* storage[ larg*sk + partial_offset ] =  X [SHARED_S];
				storage[ larg*sk + partial_offset ] +=	( ( local_st - tempoprima ) * (X1[SHARED_S]-X[SHARED_S])  ) / tau; */

				// storage[ larg*sk + partial_offset ] = X [SHARED_S];
				int bla = X1[SHARED_S]-X[SHARED_S];
				tau_t bla2 = (tau_t) bla;
				tau_t delta = (bla2*( local_st - tempoprima )) / tau;		
				storage[ larg*sk + partial_offset ] = X[SHARED_S] + (unsigned int)rint(delta);
				/*
				unsigned int s= species_to_sample_vector[sk];
				storage[ larg*DEV_CONST_SAMPLESPECIES*i + larg*sk + gid ] = X[SHARED_S];	
				*/
				// storage[ larg*DEV_CONST_SAMPLESPECIES*i + larg*sk + gid ] = X[SHARED_S] + (unsigned int)rint(delta);


			} // end for s

			i++;	

			// argh?!
			if ( i >= DEV_CONST_SAMPLES ) break;

		} // end while

		checkedsample[gid] = i;

	} // endif

}

// template <bool SSA_steps>
// extern __shared__ sh_type global_array[];

// template<int action>
__global__ void
	__launch_bounds__(64) 
	PreCalculations(
	unsigned int* X_global,			// S x thread
	param_t* A0_global,				// 1 x thread
	param_t* PAR_global,			// R x thread
	stoich_t* left,
	stoich_f* left_flatten,	
	stoich_t* var,
	stoich_f* rav,
	hor_t* hor,
	hor_t* hort,
	char* SSA,
	tau_t* tempo,
	unsigned int* prossimo_campione,
	tau_t* samples_times,
	unsigned int * samples_storage,
	unsigned int* species_to_sample_vector, 

	unsigned int* feeds_vector,
	unsigned int* feeds_values,

	g_t* G,

	
#ifdef USE_XORWOW
	curandState* cs  
#else
	curandStateMRG32k3a* cs
#endif
	, bool FORCE_SSA
	)  {


	extern __shared__ sh_type global_array[];

	unsigned int bdim = blockDim.x;

	unsigned int*	X		= (unsigned int*)global_array;	
	param_t*		PAR		= (param_t*)&X[DEV_CONST_SPECIES*bdim];				
	param_t*		PROP	= (param_t*)&PAR[DEV_CONST_REACTIONS*bdim];	
	unsigned int*   K_rule	= (unsigned int*)&PROP[DEV_CONST_REACTIONS*bdim];
	// g_t*			G		= (g_t*)&K_rule[DEV_CONST_REACTIONS*bdim];
	int*			X1		= (int*)&K_rule[DEV_CONST_REACTIONS*bdim];	
	char*			CRIT	= (char*)&X1[DEV_CONST_SPECIES*bdim];
	// tau_t*			S_INST  = (tau_t*)&CRIT[DEV_CONST_REACTIONS*bdim];


	param_t			A0  = 0.0f;
	tau_t			tau = 0.0f;

	unsigned int tid  = threadIdx.x;	
	unsigned int gid  = tid + bdim*blockIdx.x;
	unsigned int larg = bdim * gridDim.x;
	
	// simulazione finita: esci
	if ( tempo[gid]>=DEV_CONST_TIMEMAX ) {
		SSA[gid] = -1;
		return;
	}
	if ( SSA[gid]==-1 ) return;
		
	SSA[gid] = 0;
	
#ifdef USE_XORWOW
	curandState localState = cs[gid];
#else
	curandStateMRG32k3a localState = cs[gid];
#endif

	// initialize local molecular amounts
	for (unsigned int s=0; s<DEV_CONST_SPECIES; s++)	 {
		X[SHARED_S] = X_global[GLOBAL_S];	
		// printf("TH%d\t %d\n", gid, X[SHARED_S]);
		X1[SHARED_S] = 0;
	}

	// initialize local parameters and critical arrays
	for (unsigned int r=0; r<DEV_CONST_REACTIONS; r++) {
		PAR[SHARED_R] = PAR_global[GLOBAL_R];
		// CRIT_global[GLOBAL_R] = 0;				
		CRIT[SHARED_R] = 0;
		K_rule[SHARED_R] = 0;
	}

	// A0 = CalculatePropensity<DEV_FLATTEN_MATRIX>( X, PROP, PAR, left ,left_flatten);
	A0 = CalculatePropensity<DEV_FLATTEN_MATRIX_CONST>( X, PROP, PAR, left ,left_flatten);
	// A0 = CalculatePropensity<DEV_NORMAL_MATRIX>( X, PROP, PAR, left ,left_flatten);

	/*if (gid==0) {
		printf("A0=%.20f\n", A0);
	}*/
	
	// update A0
	A0_global[gid] = A0;

	// simulazione finita: esci
	if (A0==0.0f) {

			// ARGH!!
			if (prossimo_campione[gid] == 0) {
				unsigned int partial_offset = larg*DEV_CONST_SAMPLESPECIES*0 + gid; // larg*s 
				for (unsigned int sk=0; sk<DEV_CONST_SAMPLESPECIES; sk++) {
					unsigned int s= species_to_sample_vector[sk];
					samples_storage[ larg*sk + partial_offset ] =  X[SHARED_S];	 		
				}
				prossimo_campione[gid]++;
			} 


		// unsigned int partial_offset_old = larg*DEV_CONST_SPECIES*(prossimo_campione[gid]-1) + gid; 
		unsigned int partial_offset_old = larg*DEV_CONST_SAMPLESPECIES*(prossimo_campione[gid]-1) + gid; 

		// no!! devo copiare il valore fino alla fine dei campioni
		for ( unsigned int i=prossimo_campione[gid]; i<DEV_CONST_SAMPLES; i++) {			
						
			// unsigned int partial_offset = larg*DEV_CONST_SPECIES*i + gid; // larg*s 
			unsigned int partial_offset = larg*DEV_CONST_SAMPLESPECIES*i + gid; 
				
			for (unsigned int s=0; s<DEV_CONST_SAMPLESPECIES; s++) {							
				samples_storage[ larg*s + partial_offset ] = samples_storage[ larg*s + partial_offset_old ];
				// samples_storage[ larg*s + partial_offset ] = 666;
			} // end for s
		}

		tempo[gid] = DEV_CONST_TIMEMAX+0.1f;
		prossimo_campione[gid] = DEV_CONST_SAMPLES;
		SSA[gid] = -1;
		return;
	} 


// #ifdef FORCE_SSA 
	if (FORCE_SSA) {
		SSA[gid] = 1;
		return;
	}
// #endif

	
	// determiniamo reazioni critiche
	switch (DEV_FLATTEN_MATRIX) {

		case DEV_NORMAL_MATRIX: 
						
			for (unsigned int r=0; r<DEV_CONST_REACTIONS; r++) {
				for (unsigned int s=0; s<DEV_CONST_SPECIES; s++) {
					stoich_t t = left[r*DEV_CONST_SPECIES+s];
					if (t==0) continue;
					unsigned int u = X[SHARED_S] / t;
					// if ( u < nc )	CRIT_global[GLOBAL_R] = 1;			
					if ( u < nc )	CRIT[SHARED_R] = 1;			
				}	
			}
		
		break;

		case DEV_FLATTEN_MATRIX:

			for (unsigned int f=0; f<DEV_CONST_FLATTEN_LEFT; f++) {
				unsigned int r = left_flatten[f].x;
				unsigned int s = left_flatten[f].y;
				unsigned int u = X[SHARED_S] / left_flatten[f].z;
				if ((u>0) && (u<nc))	CRIT[SHARED_R] = 1;
				// if (u<nc)	CRIT[SHARED_R] = 1;
			}

		break;

	} // end switch
	

	// calcoliamo G
	for ( unsigned short s=0; s<DEV_CONST_SPECIES; s++ ) {
		hor_t tipo=hort[s];
		switch ( hor[s] )  {

			case 1: 				
				G[GLOBAL_S] = 1.0f;
			break;

			case 2:
				if ( tipo == 2 ) {					
					/* if (X[SHARED_S]==1) {
						G[GLOBAL_S] = 666;					
					} else {*/
					G[GLOBAL_S] = 2.0f + 1.0f / ( (param_t) X[SHARED_S] - 1 ); 
					// }
				} else {					
					G[GLOBAL_S] = 2.0f;
				}
			break;

			case 3:
				if ( tipo == 3 ) {		
					/* G[GLOBAL_S] = 777;
					break;
					if (X[SHARED_S]==1) {
						G[GLOBAL_S] = FLT_MAX;
						break;
					}
					if (X[SHARED_S]==2) {
						G[GLOBAL_S] = FLT_MAX;
						break;
					} */
					G[GLOBAL_S] = 3.0f + 1.0f / ( X[SHARED_S] - 1 ) + 2.0f / ( X[SHARED_S] - 2 );
				} else if ( tipo == 2 ) {					
					/*if (X[SHARED_S]==1) {
						G[GLOBAL_S] = FLT_MAX ;
						break;
					}*/
					G[GLOBAL_S] = 3.0f + 1.5f / ( X[SHARED_S] - 1 );
				} else {					
					G[GLOBAL_S] = 3.0f;
				}
			break;
			
			default: 
				G[GLOBAL_S] = 1;
			break;
		}
	}

	musigma_t tmp3, tmp4, macs, tmp = FLT_MAX;
	tau_t tau1 = FLT_MAX;
	tau_t tau2 = FLT_MAX;
	

#ifndef USE_RAV4

	/// calcoliamo mu e sigma
	for (unsigned int s=0; s<DEV_CONST_SPECIES; s++) {
		
		musigma_t mu = 0.0f;
		musigma_t sigma = 0.0f;
		// macs = 1;
		
		for (unsigned int r=0; r<DEV_CONST_REACTIONS; r++) {
			stoich_t variazione = var[r*DEV_CONST_SPECIES+s];						
			mu +=    (1-CRIT[SHARED_R]) * ( variazione * PROP[SHARED_R] );	
			sigma += (1-CRIT[SHARED_R]) * ( variazione*variazione * PROP[SHARED_R] );
		}

		if (gid==0) {
			printf("Mu: %.10f, Sigma: %.10f\n", mu, sigma);
		}

		musigma_t tmp2 = (eps * (float)( X[s*bdim+tid] )) / G[GLOBAL_S];

		/*if (gid==0) {
			printf("X= %d, G= %.10f\n",  X[s*bdim+tid], G[GLOBAL_S] );
		}*/
		
		macs =
			( tmp2 > 1.0f )?
				tmp2 : 1.0f;


		/*if (gid==0) {			
			printf("TMP2: %.10f\n", macs);
		}*/	

		/* if (tmp2 >= 1.0f )
			macs = tmp2; */
		
		if ( mu!=0 ) {			
			if ( sigma!=0 ) {
				tmp3 = macs / fabs(mu);
				tmp4 = (( macs*macs ) / sigma);
				if (tmp3 < tmp4) tmp = tmp3;
				else			 tmp = tmp4;
			}
		}
			
		/// TAU1 assign
		if ( tmp < tau1 ) tau1 = tmp;

		/*if (gid==0) {
			printf("TMP : %.10f\n", tmp);					
		}*/
		

	} // endfor 
#else

	/*
	for (unsigned int h=0; h<DEV_CONST_FLATTEN_VAR; h++ ) {
		unsigned int r= rav[h].x;
		unsigned int s= rav[h].y;
		stoich_t variazione = rav[h].z;
	}*/

	unsigned int h=0;
	unsigned int s=0;
	// unsigned int totale_giri=0;

	while (h<DEV_CONST_FLATTEN_VAR) {

		musigma_t mu = 0;
		musigma_t sigma = 0;
		
		// finché è la stessa specie...
		while(s==rav[h].y) {

			// printf("%d, %d, %d\n", rav[h].x, rav[h].y, rav[h].z);

			// totale_giri ++;
			// if (totale_giri>20) return;

			unsigned int r=rav[h].x;
			stoich_t variazione = rav[h].z;

			mu +=    (1-CRIT[SHARED_R]) * ( variazione * PROP[SHARED_R] );	
			sigma += (1-CRIT[SHARED_R]) * ( variazione*variazione * PROP[SHARED_R] );
			
			h++;
		}

		/* if (gid==0)
			printf("Mu: %.10f, Sigma: %.10f\n", mu, sigma); */

		musigma_t tmp2 = (eps * (float)( X[s*bdim+tid] )) / G[GLOBAL_S];

		
		/*if (gid==0) {
			printf("X= %d, G= %.10f\n",  X[s*bdim+tid], G[GLOBAL_S] );
		}*/
		
		macs =
			( tmp2 > 1.0f )?
				tmp2 : 1.0f;


		/* if (gid==0) {			
			printf("TMP2: %.10f\n", macs);
		}*/

		/* if (tmp2 >= 1.0f )
			macs = tmp2; */
		
		if ( mu!=0 ) {			
			if ( sigma!=0 ) {
				tmp3 = macs / fabs(mu);
				tmp4 = (( macs*macs ) / sigma);
				if (tmp3 < tmp4) tmp = tmp3;
				else			 tmp = tmp4;
			}
		}
			
		/// TAU1 assign
		if ( tmp < tau1 ) tau1 = tmp;

		s++;
		
	}

#endif


	/*if (gid==0) {
		printf("tau1-pre: %.10f\n", tau1);
		printf("10/A0: %.10f\n", 10.0f/A0);
	}*/
	

#ifdef DO_SSA_STEPS
	// tau1 è il tau calcolato da tau-leaping, A0 da "SSA": nel caso torniamo ad esso.
	if ( tau1 < 10.0f / A0 ) {
		/* if (gid==0) {
			printf("esco ad SSA\n");
		} */
		SSA[gid] = 1;
		return;
	}
#endif

	// vettore criticità vale 0 se non è critica, 1 se lo è.
	param_t a_0_c = 0.0f;
	for ( unsigned int r=0; r<DEV_CONST_REACTIONS; r++ ) {
		// a_0_c += ( PROP[SHARED_R] * CRIT_global[GLOBAL_R] );				
		// a_0_c += ( PROP[SHARED_R] * CRIT[SHARED_R] );				
		if (CRIT[SHARED_R]==1)
			a_0_c +=  PROP[SHARED_R];

	}

	/*
	if (gid==0) {
		printf("A0_c=%.20f\n", a_0_c);
	}*/
	

	// se a_0_c > 0 allora esistono delle reazioni critiche
	// tau2 è il tempo necessario a eseguirne una
	if ( a_0_c>0.0f ) {
		tmp4 = curand_uniform( &localState );
		tau2 = ( 1.0f / a_0_c ) * log( 1.0f / tmp4 );
	} 


	
	// If the time needed to execute a critical reaction
	// is longer than the time to execute all the non-critical
	// reaction, execute noncritical reactions only
	//
	// prendo il tempo inferiore tra tau-leaping e SSA
	tau = 
		tau1<tau2?
			tau1 : tau2;

	/*
	if (gid==0) {
		printf("tau1=%.20f\n", tau1);
		printf("tau2=%.20f\n", tau2);
	}

	
	if (gid==0) {
		printf("Tau_tau=%f\n", tau);
	}
	*/

	// printf("GPU conf. tempi> %f\t%f\n", tau1, tau2);
	
	// se ci mette meno la critica, estrai UNA SOLA regola critica utilizzando un passo SSA
	if ( tau1>tau2 ) {
		unsigned int r=0;		
		tmp4 = curand_uniform( &localState ) * a_0_c;
		// param_t alpha = PROP[ tid ] * CRIT_global[GLOBAL_R];
		param_t alpha = PROP[ tid ]; // * CRIT[SHARED_R];
		while( tmp4 > alpha ){			
			r++;
			// alpha += PROP[SHARED_R] * CRIT_global[GLOBAL_R];
			alpha += PROP[SHARED_R] * CRIT[SHARED_R];
		}

		K_rule[ SHARED_R ] = 1;		
		// printf("GPU critic step> scelta reaz %d\n", r);
	}

	

	
	for ( unsigned int r=0; r<DEV_CONST_REACTIONS; r++) {
		// if ( CRIT_global[GLOBAL_R] == 0 ) {
		if ( CRIT[SHARED_R] == 0 ) {
			// K_rule[SHARED_R] = poisson_rejection<USE_NORMAL>( cs, tau* PROP[SHARED_R]);
			// K_rule[SHARED_R] = poisson_rejection<USE_CURAND>( cs, tau* PROP[SHARED_R]);// 
			// unsigned int puass = curand_poisson( &localState, tau* PROP[SHARED_R] );		
			// if (tid==0) printf("Puass: %.20f\n", puass);
			K_rule[SHARED_R] = curand_poisson( &localState, tau* PROP[SHARED_R] );		;
			// K_rule[SHARED_R] = poisson_rejection<USE_POISSON>( cs, tau* PROP[SHARED_R]);
		/*} else {
			K_rule[SHARED_R] = 0; */
		}
	} 

	// Extract the rules from a poissonian distribution 
	// In questo caso è meglio effettuare un branch, perché il calcolo della poissoniana 
	// porta via tantissimo 
	// resta da aggiornare lo stato
	while (!(test_update( var )) ) {				

		// break;
		
		tau *= 0.5f;
		
		for ( unsigned int r=0; r<DEV_CONST_REACTIONS; r++) {
			// if ( CRIT_global[GLOBAL_R] == 0 ) {
			if ( CRIT[SHARED_R] == 0 ) {
			    // K_rule[SHARED_R] = poisson_rejection<USE_NORMAL>( cs, tau* PROP[SHARED_R]);
				// K_rule[SHARED_R] = poisson_rejection<USE_CURAND>( cs, tau* PROP[SHARED_R]);// 
				unsigned int puass = curand_poisson( &localState, tau* PROP[SHARED_R] );		
				// if (tid==0) printf("Puass: %.20f\n", puass);
				K_rule[SHARED_R] = puass;
				// K_rule[SHARED_R] = poisson_rejection<USE_POISSON>( cs, tau* PROP[SHARED_R]);
			} else {
				K_rule[SHARED_R] = 0;
			}
		} 


	}; // end while

	cs[gid] = localState;

	/*if(gid==0){
		for (unsigned int r=0; r<DEV_CONST_REACTIONS; r++ )
			printf("%d\t", K_rule[SHARED_R]);
		printf("\n");
	}
*/
	
	//check_interpolated_subfitness_sm( tempo, tau, &prox_campione, X_shared, X_1_shared, specie_fitness, subfitness, indice_thread );
	//check_interpolated_subfitness_sm( tempo, tau, &prox_campione, X_shared, X_1_shared, subfitness, indice_thread );

#ifdef COLLECT_DYNAMICS
	check_time_instant( tempo[gid], tau, samples_times, X, X1, prossimo_campione, samples_storage, species_to_sample_vector );
#endif

	for ( unsigned int s=0; s<DEV_CONST_SPECIES; s++ ) {
		X_global[GLOBAL_S] = X1[SHARED_S];	 
	}

	/* FEED (NEW!!) */
#ifndef DISABLE_FEED
	for (unsigned int u=0; u<DEV_CONST_FEEDSPECIES; u++ ) {
		X_global[ feeds_vector[u] * larg + gid ] = feeds_values[ u * larg + tid ] ;		
		// X_global[ feeds_vector[u] * larg + tid ] = 12345;		
		// if (gid==0) printf("specie feed %d, pos %d\n", u,feeds_vector[u] );
	}
#endif

	// printf("TAU-LEA%.20f\n", tau);

	// update tempo reazione
	tempo[gid] += tau;

	// if (gid==0) printf("T=%f.\n", tempo[gid]+tau);

	// if (gid==0) printf("TEMPO TAU-LP: %f\n", tempo[0]);

	// tutto okay, tau-leaping eseguito
	return;

}




// template<int action>
__global__ void EventualSSASteps( 
	const unsigned int steps, 
	unsigned int* X_global, 
	const param_t* PAR_global,
	const stoich_t * left,
	const stoich_f * left_flatten, 
	const stoich_t* var,
	tau_t* tempo, 
	char* SSA,
	unsigned int* prossimo_campione,
	tau_t* samplestimes,
	unsigned int * storage,
	unsigned int* species_to_sample_vector, 

	unsigned int* feeds_vector,
	unsigned int* feeds_values,

	
#ifdef USE_XORWOW
	curandState* cs  
#else
	curandStateMRG32k3a* cs
#endif
	
	) {

	extern __shared__ sh_type global_array[];

	unsigned int tid = threadIdx.x;	
	unsigned int gid = threadIdx.x+blockIdx.x*blockDim.x;
	unsigned int bdim = blockDim.x;
	unsigned int larg = bdim * gridDim.x;

	if (SSA[gid]!=1)	return;

	unsigned int*	X		= (unsigned int*)global_array;	
	param_t*		PAR		= (param_t*)&X[DEV_CONST_SPECIES*bdim];				
	param_t*		PROP	= (param_t*)&PAR[DEV_CONST_REACTIONS*bdim];	
	// tau_t*			S_INST  = (tau_t*)&PROP[DEV_CONST_REACTIONS*bdim];

#ifdef USE_XORWOW
	curandState localState = cs[gid];
#else 
	curandStateMRG32k3a localState = cs[gid];
#endif 

	tau_t tau = 0.0f;
	param_t A0;

	for (unsigned int s=0; s<DEV_CONST_SPECIES; s++) {
		X[SHARED_S] = X_global[GLOBAL_S];	
	}

	for (unsigned int r=0; r<DEV_CONST_REACTIONS; r++) {
		PAR[SHARED_R] = PAR_global[GLOBAL_R];		
	}

	// if (gid==0) printf("T=%f.\n", tempo[gid]+tau);

	for (unsigned int v=0; v<steps; v++) {

	

		A0 = CalculatePropensity<DEV_FLATTEN_MATRIX_CONST>( X, PROP, PAR, left, left_flatten);
				
		// simulazione finita, setta il tempo alla fine ed esci
		if (A0 == 0){

			// ARGH!!
			if (prossimo_campione[gid] == 0) {
				unsigned int partial_offset = larg*DEV_CONST_SAMPLESPECIES*0 + gid; // larg*s 
				for (unsigned int sk=0; sk<DEV_CONST_SAMPLESPECIES; sk++) {
					unsigned int s= species_to_sample_vector[sk];
					storage[ larg*sk + partial_offset ] =  X[SHARED_S];	 		
				}
				prossimo_campione[gid]++;
			} 

			// unsigned int partial_offset_old = larg*DEV_CONST_SPECIES*(prossimo_campione[gid]-1) + gid; // larg*s 
			unsigned int partial_offset_old = larg*DEV_CONST_SAMPLESPECIES*(prossimo_campione[gid]-1) + gid; // larg*s 

			// no!! devo copiare il valore fino alla fine dei campioni
			for ( unsigned int i=prossimo_campione[gid]; i<DEV_CONST_SAMPLES; i++) {			
						
				// unsigned int partial_offset = larg*DEV_CONST_SPECIES*i + gid; // larg*s 
				unsigned int partial_offset = larg*DEV_CONST_SAMPLESPECIES*i + gid; // larg*s 
	
				for (unsigned int s=0; s<DEV_CONST_SAMPLESPECIES; s++) {							
					// storage[ larg*s + partial_offset ] = 666;
					storage[ larg*s + partial_offset ] = storage[ larg*s + partial_offset_old ];
				} // end for s
			}

			tempo[gid] = DEV_CONST_TIMEMAX + 0.1f;
			prossimo_campione[gid] = DEV_CONST_SAMPLES;
			SSA[gid] = -1;
			return;
		}
				
		param_t rnd =  curand_uniform( &localState ) * A0;
		param_t rnd_t =  curand_uniform( &localState );
		
		unsigned int r = 0;
		param_t alpha = PROP[SHARED_R];

		while( rnd > alpha ) {
			r++;
			alpha += PROP[SHARED_R];
		}		
			
		tau +=  (1.0f/A0)*log(1.0f/rnd_t);
				
#ifdef COLLECT_DYNAMICS

		/* CHECK PUNTO */
		unsigned int i = prossimo_campione[gid];


		if ( i<DEV_CONST_SAMPLES ) {

			// per tutti gli istanti coperti dal salto temporale
			while( tempo[gid] + tau > samplestimes[i] ) {
			
				for (unsigned int sk=0; sk<DEV_CONST_SAMPLESPECIES; sk++) {
					unsigned int s= species_to_sample_vector[sk];
					storage[ larg*DEV_CONST_SAMPLESPECIES*i + larg*sk + gid ] = X[SHARED_S];	 		
				} // end for s

				i++;	

				// argh?!
				if ( i >= DEV_CONST_SAMPLES ) break;

			} // end while

			prossimo_campione[gid] = i;

		} // endif

#endif
		
#ifdef USE_FEAR_SSA

		for (unsigned int s=0; s<DEV_CONST_SPECIES; s++) {			
			X[SHARED_S] += DEV_CONST_MAT_FEAR[r*DEV_CONST_SPECIES+s];	
		}

#else 

		/* aggiorna quantità */	
		for (unsigned int s=0; s<DEV_CONST_SPECIES; s++) {
			X[SHARED_S] += var[r*DEV_CONST_SPECIES+s];			
		}

		/* FEED (NEW!!) */
#ifndef DISABLE_FEED
		for (unsigned int u=0; u<DEV_CONST_FEEDSPECIES; u++ ) {
			// X_global[ feeds_vector[u] * bdim + tid ] = feeds_values[ u * bdim + tid ];
			X[feeds_vector[u]*bdim+tid] = feeds_values[ u * larg + tid ];
		}
#endif

#endif

	}

	// dump quantità su quelle in memoria globale
	for (unsigned int s=0; s<DEV_CONST_SPECIES; s++) {
		X_global[GLOBAL_S] = X[SHARED_S];
	}

	cs[gid] = localState;
	tempo[gid] += tau;

}





// template<int action>
__global__ void LightweightSSA( 
	const unsigned int steps, 
	unsigned int* X_global, 
	const param_t* PAR_global,
	const stoich_t * left,
	const stoich_f * left_flatten, 
	const stoich_t* var,
	tau_t* tempo, 
	char* SSA,
	unsigned int* prossimo_campione,
	tau_t* samplestimes,
	unsigned int * storage,
	unsigned int* species_to_sample_vector, 

	unsigned int* feeds_vector,
	unsigned int* feeds_values,

	
#ifdef USE_XORWOW
	curandState* cs  
#else
	curandStateMRG32k3a* cs
#endif
	
	) {

	extern __shared__ sh_type global_array[];

	unsigned int tid = threadIdx.x;	
	unsigned int gid = threadIdx.x+blockIdx.x*blockDim.x;
	unsigned int bdim = blockDim.x;
	unsigned int larg = bdim * gridDim.x;

	if (SSA[gid]!=1)	return;

	unsigned int*	X		= (unsigned int*)global_array;	
	param_t*		PROP = (param_t*)&X[DEV_CONST_SPECIES*bdim];				
	// param_t*		PROP	= (param_t*)&PAR[DEV_CONST_REACTIONS*bdim];	
	// tau_t*			S_INST  = (tau_t*)&PROP[DEV_CONST_REACTIONS*bdim];

#ifdef USE_XORWOW
	curandState localState = cs[gid];
#else 
	curandStateMRG32k3a localState = cs[gid];
#endif 

	tau_t tau = 0.0f;
	param_t A0;

	for (unsigned int s=0; s<DEV_CONST_SPECIES; s++) {
		X[SHARED_S] = X_global[GLOBAL_S];	
	}

	/*
	for (unsigned int r=0; r<DEV_CONST_REACTIONS; r++) {
		PAR[SHARED_R] = PAR_global[GLOBAL_R];		
	}
	*/

	if (gid==0) printf("T=%f.\n", tempo[gid]+tau);

	for (unsigned int v=0; v<steps; v++) {
		
		A0 = CalculatePropensity<DEV_FLATTEN_MATRIX_CONST_GLOBPARAMS>( X, PROP, PAR_global, left, left_flatten);
				
		// simulazione finita, setta il tempo alla fine ed esci
		if (A0 == 0){

			// ARGH!!
			if (prossimo_campione[gid] == 0) {
				unsigned int partial_offset = larg*DEV_CONST_SAMPLESPECIES*0 + gid; // larg*s 
				for (unsigned int sk=0; sk<DEV_CONST_SAMPLESPECIES; sk++) {
					unsigned int s= species_to_sample_vector[sk];
					storage[ larg*sk + partial_offset ] =  X[SHARED_S];	 		
				}
				prossimo_campione[gid]++;
			} 

			// unsigned int partial_offset_old = larg*DEV_CONST_SPECIES*(prossimo_campione[gid]-1) + gid; // larg*s 
			unsigned int partial_offset_old = larg*DEV_CONST_SAMPLESPECIES*(prossimo_campione[gid]-1) + gid; // larg*s 

			// no!! devo copiare il valore fino alla fine dei campioni
			for ( unsigned int i=prossimo_campione[gid]; i<DEV_CONST_SAMPLES; i++) {			
						
				// unsigned int partial_offset = larg*DEV_CONST_SPECIES*i + gid; // larg*s 
				unsigned int partial_offset = larg*DEV_CONST_SAMPLESPECIES*i + gid; // larg*s 
	
				for (unsigned int s=0; s<DEV_CONST_SAMPLESPECIES; s++) {							
					// storage[ larg*s + partial_offset ] = 666;
					storage[ larg*s + partial_offset ] = storage[ larg*s + partial_offset_old ];
				} // end for s
			}

			tempo[gid] = DEV_CONST_TIMEMAX + 0.1f;
			prossimo_campione[gid] = DEV_CONST_SAMPLES;
			SSA[gid] = -1;
			return;
		}
				
		param_t rnd =  curand_uniform( &localState ) * A0;
		param_t rnd_t =  curand_uniform( &localState );
		
		unsigned int r = 0;
		param_t alpha = PROP[SHARED_R];

		while( rnd > alpha ) {
			r++;
			alpha += PROP[SHARED_R];
		}		
			
		tau +=  (1.0f/A0)*log(1.0f/rnd_t);
				
#ifdef COLLECT_DYNAMICS

		/* CHECK PUNTO */
		unsigned int i = prossimo_campione[gid];


		if ( i<DEV_CONST_SAMPLES ) {

			// per tutti gli istanti coperti dal salto temporale
			while( tempo[gid] + tau > samplestimes[i] ) {
			
				for (unsigned int sk=0; sk<DEV_CONST_SAMPLESPECIES; sk++) {
					unsigned int s= species_to_sample_vector[sk];
					storage[ larg*DEV_CONST_SAMPLESPECIES*i + larg*sk + gid ] = X[SHARED_S];	 		
				} // end for s

				i++;	

				// argh?!
				if ( i >= DEV_CONST_SAMPLES ) break;

			} // end while

			prossimo_campione[gid] = i;

		} // endif

#endif
		
#ifdef USE_FEAR_SSA

		for (unsigned int s=0; s<DEV_CONST_SPECIES; s++) {			
			X[SHARED_S] += DEV_CONST_MAT_FEAR[r*DEV_CONST_SPECIES+s];	
		}

#else 

		/* aggiorna quantità */	
		for (unsigned int s=0; s<DEV_CONST_SPECIES; s++) {
			X[SHARED_S] += var[r*DEV_CONST_SPECIES+s];			
		}

		/* FEED (NEW!!) */
#ifndef DISABLE_FEED
		for (unsigned int u=0; u<DEV_CONST_FEEDSPECIES; u++ ) {
			// X_global[ feeds_vector[u] * bdim + tid ] = feeds_values[ u * bdim + tid ];
			X[feeds_vector[u]*bdim+tid] = feeds_values[ u * larg + tid ];
		}
#endif

#endif

	}

	// dump quantità su quelle in memoria globale
	for (unsigned int s=0; s<DEV_CONST_SPECIES; s++) {
		X_global[GLOBAL_S] = X[SHARED_S];
	}

	cs[gid] = localState;
	tempo[gid] += tau;

}



__global__ void ReduceSimulationsCompleted_old(	char* SSA ) {

	extern __shared__ sh_type global_array[];

	unsigned int tid = threadIdx.x;
	unsigned int gid = threadIdx.x + blockDim.x * blockIdx.x;

	global_array[tid] = SSA[gid];

	__syncthreads();
	
	for (unsigned int s=blockDim.x/2; s>0; s>>=1) {
		if (tid < s) {
			global_array[tid] = max( global_array[tid], global_array[tid + s]);
		}
		__syncthreads();
	}

	// metto in SSA alla posizione "#blocco" lo stato di terminazione
	if (tid==0)	{
		SSA[blockIdx.x]=global_array[0];
		// printf("%d:%d\n", gid, global_array[0]);
	}


}



__global__ void ReduceSimulationsCompleted(char* SSA, char* ended) {
	
	extern __shared__ sh_type sdata[];
	unsigned int tid = threadIdx.x;
	unsigned int bdim = blockDim.x;
	unsigned int gid = threadIdx.x + blockDim.x * blockIdx.x;
	
	sdata[tid] = SSA[gid]; // assign initial value
	__syncthreads();

	// do reduction in shared mem. this example assumes
	// that the block size is 256; see the “reduction”
	// sample in the GPU Computing SDK for a complete and
	// general implementation
	for (unsigned int s=bdim/2; s>32; s>>=1) 
	{
		if (tid < s)
		sdata[tid] += sdata[tid + s];
		__syncthreads();
	}
	// no __syncthreads() necessary after each of the
	// following lines as long as we access the data via
	// a pointer declared as volatile because the 32 threads
	// in each warp execute in lock-step with each other
	volatile char *smem = sdata;
	smem[tid] += smem[tid + 32];
	smem[tid] += smem[tid + 16];
	smem[tid] += smem[tid + 8];
	smem[tid] += smem[tid + 4];
	smem[tid] += smem[tid + 2];
	smem[tid] += smem[tid + 1];
	
	// write result for this block to global mem 
	if (tid==0)	{
		if (smem[0]== -blockDim.x) 
			ended[blockIdx.x] = -1;
		else
			ended[blockIdx.x] = 1;
		// printf("Risultato reduce:%d\n", ended[blockIdx.x]);
	}

}


template <unsigned int blockSize>
__device__ void warpReduce(volatile int *sdata, unsigned int tid) {
	if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
	if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
	if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
	if (blockSize >= 8) sdata[tid] += sdata[tid + 4];
	if (blockSize >= 4) sdata[tid] += sdata[tid + 2];
	if (blockSize >= 2) sdata[tid] += sdata[tid + 1];
}




/*
    This version adds multiple elements per thread sequentially.  This reduces the overall
    cost of the algorithm while keeping the work complexity O(n) and the step complexity O(log n).
    (Brent's Theorem optimization)

    Note, this kernel needs a minimum of 64*sizeof(T) bytes of shared memory.
    In other words if blockSize <= 32, allocate 64*sizeof(T) bytes.
    If blockSize > 32, allocate blockSize*sizeof(T) bytes.
*/
/*
template <unsigned int blockSize>
__global__ void
reduce7(char *g_idata, int *g_odata, unsigned int n)
{
     // *sdata = SharedMemory<T>();
	extern __shared__ sh_type global_array[];
	int* sdata = (int*)global_array;


    // perform first level of reduction,
    // reading from global memory, writing to shared memory
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockSize*2 + threadIdx.x;
    unsigned int gridSize = blockSize*2*gridDim.x;

    int mySum = 0;

    // we reduce multiple elements per thread.  The number is determined by the
    // number of active thread blocks (via gridDim).  More blocks will result
    // in a larger gridSize and therefore fewer elements per thread
    while (i < n)
    {
        mySum += g_idata[i];

        // ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
        if (i + blockSize < n)
            mySum += g_idata[i+blockSize];

        i += gridSize;
    }

    // each thread puts its local sum into shared memory
    sdata[tid] = mySum;
    __syncthreads();


    // do reduction in shared mem
    if (blockSize >= 512)
    {
        if (tid < 256)
        {
            sdata[tid] = mySum = mySum + sdata[tid + 256];
        }

        __syncthreads();
    }

    if (blockSize >= 256)
    {
        if (tid < 128)
        {
            sdata[tid] = mySum = mySum + sdata[tid + 128];
        }

        __syncthreads();
    }

    if (blockSize >= 128)
    {
        if (tid <  64)
        {
            sdata[tid] = mySum = mySum + sdata[tid +  64];
        }

        __syncthreads();
    }

    if (tid < 32)
    {
        // now that we are using warp-synchronous programming (below)
        // we need to declare our shared memory volatile so that the compiler
        // doesn't reorder stores to it and induce incorrect behavior.
        volatile int *smem = sdata;

        if (blockSize >=  64)
        {
            smem[tid] = mySum = mySum + smem[tid + 32];
        }

        if (blockSize >=  32)
        {
            smem[tid] = mySum = mySum + smem[tid + 16];
        }

        if (blockSize >=  16)
        {
            smem[tid] = mySum = mySum + smem[tid +  8];
        }

        if (blockSize >=   8)
        {
            smem[tid] = mySum = mySum + smem[tid +  4];
        }

        if (blockSize >=   4)
        {
            smem[tid] = mySum = mySum + smem[tid +  2];
        }

        if (blockSize >=   2)
        {
            smem[tid] = mySum = mySum + smem[tid +  1];
        }
    }

    // write result for this block to global mem
    if (tid == 0)
        g_odata[blockIdx.x] = sdata[0];
}





template <unsigned int blockSize>
__global__ void reduce6(char *g_idata, int *g_odata, unsigned int n) {
	// extern __shared__ char sdata[];
	extern __shared__ sh_type global_array[];
	int* sdata = (int*)global_array;

	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*(blockSize*2) + tid;
	unsigned int gridSize = blockSize*2*gridDim.x;
	sdata[tid] = 0;
	while (i < n) { 
		sdata[tid] += (int)g_idata[i] + (int)g_idata[i+blockSize]; 
		i += gridSize; 
	}
	__syncthreads();
	if (blockSize >= 512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
	if (blockSize >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
	if (blockSize >= 128) { if (tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads(); }
	if (tid < 32) warpReduce<blockSize>(sdata, tid);
	if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

__global__ void ReduceSimulationsCompleted_somma(	char* SSA, char* ended ) {

	unsigned int tid = threadIdx.x;
	unsigned int gid = threadIdx.x + blockDim.x * blockIdx.x;

	extern __shared__ sh_type global_array[];

	global_array[tid] = SSA[gid];

	__syncthreads();
	
	for (unsigned int s=blockDim.x/2; s>0; s>>=1) {
		if (tid < s) {
			global_array[tid] +=  global_array[tid + s];
		}
		__syncthreads();
	}

	// metto in SSA alla posizione "#blocco" lo stato di terminazione
	if (tid==0)	{
		if (global_array[0]== -blockDim.x) 
			// SSA[blockIdx.x] = -1;
			ended[blockIdx.x] = -1;
		else
			//SSA[blockIdx.x] = 1;
			ended[blockIdx.x] = 1;
		// printf("%d:%d\n", gid, global_array[0]);
	}


}

*/



void ReadTempi( unsigned int specie, unsigned int reazioni, unsigned int parallel_threads, unsigned int parallel_blocks, bool dump = false) {


	cudaThreadSynchronize();

	cudaMemcpy( host_T, dev_perthread_T, sizeof(tau_t)*parallel_threads*parallel_blocks, cudaMemcpyDeviceToHost );
	CudaCheckError();

	float old_t = host_T[0];


	for (unsigned int i=0; i<parallel_blocks; i++) {
		for (unsigned int t=0; t<parallel_threads; t++) {
			printf("Time thread%d, block %d: %.30f\n", t,i,host_T[ ACCESS_SINGLE ]);
			if (old_t!=host_T[ ACCESS_SINGLE ]) {
				old_t =host_T[ ACCESS_SINGLE ];
				//printf("DIVERSI!\n");
				printf("");
			}
		}
	}
	printf("\n");
}


void ReadDataBack(unsigned int specie, unsigned int reazioni,unsigned int parallel_threads, unsigned int pblocks, bool dump=false) {

	cudaThreadSynchronize();	
		
		cudaMemcpy( host_X,  dev_perthread_X,  sizeof(unsigned int)*specie*parallel_threads*pblocks, cudaMemcpyDeviceToHost );
		CudaCheckError();
		cudaMemcpy( host_T,  dev_perthread_T,  sizeof(tau_t)*parallel_threads*pblocks, cudaMemcpyDeviceToHost );
		CudaCheckError();
#ifdef SERVICEDUMP 
		cudaMemcpy( host_A0, dev_perthread_A0, sizeof(param_t)*parallel_threads*pblocks, cudaMemcpyDeviceToHost );
		CudaCheckError();
		cudaMemcpy( host_SSA,dev_perthread_SSA,sizeof(char)*parallel_threads*pblocks, cudaMemcpyDeviceToHost);
		CudaCheckError(); 
#endif 
		

		for (unsigned int i=0; i<pblocks; i++) {
			for (unsigned int t=0; t<parallel_threads; t++) {
			
				printf("   Block %d, thread %d\n", i, t);

				for (unsigned int s=0; s<specie; s++) {					
					printf("%d\t", host_X[ ACCESS_SPECIES ]); 
				}
				printf ("T: %f\t", host_T[ ACCESS_SINGLE ]);
#ifdef SERVICEDUMP
				printf ("A0: %f", host_A0[ ACCESS_SINGLE ] );  				

				if ( host_SSA[ACCESS_SINGLE] != -1 ) printf(" ---------------- ");

				printf ("\nFlag: %d", host_SSA[ACCESS_SINGLE]); 
				printf("\n\n");

#endif 

			}
		}
		printf ("\n");
	
}

void WriteData(float interv, std::string d, unsigned int specie, unsigned int reazioni, unsigned int parallel_threads, unsigned int pblocks) {

	static float posizione = 0;
	static int   step = 0;

	posizione += interv;

	unsigned int i=0, t=0;

	if ( posizione>interv*step ) {
		step++;
		std::ofstream myfile;
		myfile.open (d.c_str(), std::fstream::app);
		myfile << host_T[0];
		
		for (unsigned s=0; s<specie; s++) {
			myfile << "\t" << host_X[ACCESS_SPECIES];  
		}

		myfile << std::endl;

		myfile.close();
	}
}


void WriteDynamics( std::string d, unsigned int parallel_threads, unsigned int pblocks, unsigned int samples, unsigned int species, unsigned int samplespecies ) {

	cudaThreadSynchronize();

	unsigned int larg =  parallel_threads * pblocks;

	unsigned int bytesize =  sizeof(unsigned int ) * larg * samples * samplespecies ;

	std::ofstream myfile;
	std::ofstream myfile_single;
	myfile.open (d.c_str());
	// myfile_single.open( d+"_single" );

	unsigned int * host_dynamics = (unsigned int *) malloc (bytesize);
	memset(host_dynamics, 0, bytesize);
	cudaMemcpy( host_dynamics, dev_perthread_storage, bytesize, cudaMemcpyDeviceToHost);
	CudaCheckError();

	for (unsigned int gid=0; gid<parallel_threads*pblocks; gid++) {		
		for (unsigned int i=0; i<samples; i++ ) {			
			myfile << host_sampletime[i] << "\t" << gid; 
			for (unsigned int s=0; s<samplespecies; s++) {
				myfile << "\t" << host_dynamics[ larg*samplespecies*i + larg*s + gid ];
			}
			myfile << "\n";
			/*
			if (gid==1) {
				myfile_single << host_sampletime[i] << "\t" << gid; 
				for (unsigned int s=0; s<species; s++) {
					myfile_single << "\t" << host_dynamics[ GLOBAL_SAMPLES ];
				}
				myfile_single<< "\n";
			} */
		}
		myfile << "\n";
	}
	myfile.close();
	// myfile_single.close();

}



void deallocate_space( unsigned int threads, unsigned int blocks, unsigned int species, unsigned int reactions, unsigned int samples, unsigned int GPU ) {
	
	// printf(" * Deallocating space on host and devices for %d threads, %d blocks, %d species, %d reactions, %d samples\n", threads, blocks, species, reactions, samples);

	free(host_X);
	free(host_T);
	free(host_A0);
	free(host_SSA);

	cudaFree(dev_perthread_X);
	CudaCheckError();

	cudaFree( dev_perthread_T );
	CudaCheckError();
	
	cudaFree( dev_perthread_A0 );
	CudaCheckError();

	cudaFree( dev_perthread_G );
	CudaCheckError();

	cudaFree( dev_perthread_SSA );
	CudaCheckError();
	
	cudaFree( dev_perthread_CRIT );
	CudaCheckError();
		
	cudaFree( dev_perthread_checkedsample );
	CudaCheckError();

	cudaFree( dev_perthread_storage );
	CudaCheckError();

	cudaFree( dev_species_to_sample_vector );
	CudaCheckError();

}

void allocate_space( unsigned int threads, unsigned int blocks, unsigned int species, unsigned int reactions, unsigned int samples, unsigned int GPU, unsigned int samplespecies, bool verbose ) {

	size_t avail, total;

	if (verbose) printf(" * Allocating space on host and devices for %d threads, %d blocks, %d species, %d reactions, %d samples\n", threads, blocks, species, reactions, samples);

	host_X =    (unsigned int*) malloc ( sizeof(unsigned int)*threads*blocks*species);
	host_T =    (tau_t*) malloc ( sizeof(tau_t) * threads*blocks );
	host_A0 =   (param_t*) malloc ( sizeof(param_t) * threads*blocks );
	host_SSA =  (char*) malloc( sizeof(char) * threads*blocks);

	cudaMalloc( &dev_perthread_X,    sizeof(unsigned int)*threads*blocks*species );
		cudaMemset( dev_perthread_X,	0, sizeof(unsigned int)*threads*blocks*species );
	CudaCheckError();

	verifica = (char*) malloc( blocks*sizeof(char));

	unsigned int total_allocated = 0;

	// cudaMalloc( &dev_perthread_PAR,  sizeof(param_t)*threads*blocks*reactions );
	cudaMalloc( &dev_perthread_T,    sizeof(tau_t)*threads*blocks );
	CudaCheckError();
	cudaMemGetInfo( &avail, &total );
	if (verbose) printf(" * Allocating %d bytes for T\n", sizeof(tau_t)*threads*blocks );
	total_allocated += sizeof(tau_t)*threads*blocks;

	cudaMalloc( &dev_perthread_A0,   sizeof(param_t)*threads*blocks );
	CudaCheckError();
	cudaMemGetInfo( &avail, &total );
	if (verbose) printf(" * Allocating %d bytes for A0\n",  sizeof(param_t)*threads*blocks );
	total_allocated += sizeof(param_t)*threads*blocks;

	// cudaMalloc( &dev_perthread_PROP, sizeof(param_t)*threads*blocks*reactions );
	cudaMalloc( &dev_perthread_G,	 sizeof(g_t)*threads*blocks*species );
	CudaCheckError();
	cudaMemGetInfo( &avail, &total );
	if (verbose) printf(" * Allocating %d bytes for G\n",  sizeof(g_t)*threads*blocks*species );
	total_allocated +=  sizeof(g_t)*threads*blocks*species ;

	cudaMalloc( &dev_perthread_SSA,  sizeof(char)*threads*blocks );
	CudaCheckError();
	cudaMemGetInfo( &avail, &total );
	if (verbose) printf(" * Allocating %d bytes for SSA\n", sizeof(char)*threads*blocks );
	total_allocated += sizeof(char)*threads*blocks;

	cudaMalloc( &dev_perthread_CRIT, sizeof(char)*threads*blocks*reactions );
	CudaCheckError();
	cudaMemGetInfo( &avail, &total );
	if (verbose) printf(" * Allocating %d bytes for SSA\n", sizeof(char)*threads*blocks*reactions );
	total_allocated += sizeof(char)*threads*blocks*reactions;

	cudaMalloc( &dev_perthread_G,    sizeof(g_t)*threads*blocks*species );
	CudaCheckError();
	cudaMemGetInfo( &avail, &total );
	if (verbose) printf(" * Allocating %d bytes for G\n", sizeof(g_t)*threads*blocks*species );
	total_allocated +=  sizeof(g_t)*threads*blocks*species;

	cudaMalloc( &dev_perthread_checkedsample,    sizeof(unsigned int)*threads*blocks );
	CudaCheckError();
	cudaMemGetInfo( &avail, &total );
	if (verbose) printf(" * Allocating %d bytes for checked samples\n", sizeof(unsigned int)*threads*blocks );
	total_allocated +=  sizeof(unsigned int)*threads*blocks;

	// alloco vettore "ho finito"
	cudaMalloc( &dev_perblock_ended, sizeof(int) * blocks );
	total_allocated += sizeof(int)*blocks;

	cudaMalloc( &dev_perthread_storage, sizeof(unsigned int )*threads*blocks*samplespecies*samples );	
	cudaMemGetInfo( &avail, &total );
	if (verbose) std::cout << " * Used " << total-avail << "/" << total << " bytes on GPU" << GPU << "\n";
	if (verbose) std::cout << " * Total allocated: " << total_allocated << "bytes"<< std::endl;
	
	CudaCheckError();
}

__global__ void setup_random ( 
	
	
#ifdef USE_XORWOW
	curandState* state  
#else
	curandStateMRG32k3a* state
#endif
	
	, unsigned long seed )
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;	
	curand_init ( seed, id, 0, &state[id] );
	// curand_init ( 0, 0, 0, &state[id] );
}

__global__ void ParallelVoting ( tau_t* dev_perthread_T, unsigned int* dev_result_voting ) {
	// __shared__ unsigned int totale;
		
	int id = blockIdx.x * blockDim.x + threadIdx.x;	

	if (id==0) *dev_result_voting = 0;
	__syncthreads();

	int res = __syncthreads_count ( dev_perthread_T[id]>=DEV_CONST_TIMEMAX );
	if (threadIdx.x==0) 
		atomicAdd(dev_result_voting, res);

}

/*
unsigned int* distribute_work( unsigned int tpb, unsigned int bpt, int* gpus) {
	
	cudaDeviceProp devProp;
	cudaGetDeviceProperties(&devProp, 0);

	int GPUcount;
	cudaGetDeviceCount(&GPUcount);
	
#ifdef DISABLE_MULTI_GPU
	GPUcount = 1;
#endif

	printf(" * %d GPUs on this system\n", GPUcount);
	*gpus = GPUcount;

	static unsigned int* jobs_array = (unsigned int*) malloc ( sizeof(unsigned int) * GPUcount );
	memset(jobs_array, 0, sizeof(unsigned int) * GPUcount );

	std::vector<unsigned int> capacity;
	for (int d=0; d<GPUcount; d++) {
		cudaSetDevice(d);
		printf(" * Streaming multiprocessors on GPU %d: %d\n", d, devProp.multiProcessorCount );
		capacity.push_back(devProp.multiProcessorCount);
	}
	cudaSetDevice(0);

	while ( bpt>0 ) {
		for (unsigned int d=0; d<GPUcount; d++) {			
			if ( devProp.multiProcessorCount<bpt ) {
				jobs_array[d] += devProp.multiProcessorCount;
				bpt -= devProp.multiProcessorCount;
			} else {
				jobs_array[d] += bpt;
				bpt = 0;
			}
		}
	}

	printf(" * Distribution of jobs on the GPUs:\n");
	for (unsigned int d=0; d<GPUcount; d++) {
		printf("   %d blocks on GPU %d\n", jobs_array[d], d);
	}

	return jobs_array;

}
*/

void check_shared_memory(unsigned int totale_bytes, unsigned int gpu, bool verbose) {

	if (verbose) printf(" * Estimated shared memory requirements: %d bytes\n", totale_bytes);

	cudaDeviceProp devProp;
    cudaGetDeviceProperties(&devProp, gpu);

	if (devProp.sharedMemPerBlock < totale_bytes ) {
		printf("ERROR: not enough memory to load this model, only %d bytes \n", devProp.sharedMemPerBlock);
		// std::cin.get();
		exit(-1);
	} else {
		if (verbose) printf(" * Shared memory requirement fulfilled by GPU%d\n", gpu);
	}

}

void check_constant_memory(unsigned int totale_bytes, bool verbose) {

	if (verbose) printf(" * Estimate constant memory requirements: %d bytes\n", totale_bytes);

	
	if (65536 < totale_bytes ) {
		printf("ERROR: not enough constant memory to load this model, only 65536 bytes \n");
		// std::cin.get();
		exit(-1);
	} else {
		if (verbose) {
			printf(" * Constant memory requirement fulfilled\n");
			if (8192<totale_bytes) printf("WARNING: Constant memory exceeded\n");		
		}
	}


}

void esci()  {

	// system("pause");
	exit(0);
}

std::string get_date_dump() {
	
	time_t rawtime;
	struct tm * timeinfo;
	time ( &rawtime );
	timeinfo = localtime ( &rawtime );
	std::stringstream dataora;
	dataora << std::setw( 2 ) << std::setfill( '0' ) << timeinfo->tm_year;
	dataora << std::setw( 2 ) << std::setfill( '0' ) << timeinfo->tm_mon;
	dataora << std::setw( 2 ) << std::setfill( '0' ) << timeinfo->tm_mday;
	dataora << std::setw( 2 ) << std::setfill( '0' ) << timeinfo->tm_hour;
	dataora << std::setw( 2 ) << std::setfill( '0' ) << timeinfo->tm_min;
	dataora << std::setw( 2 ) << std::setfill( '0' ) << timeinfo->tm_sec;
	std::string dataoradump = dataora.str();

	return dataoradump;

}

std::string get_output_folder(int argc, const char* argv, bool verbose=false) {


	std::string output_folder = "./";
	if (argc>3) {
		output_folder = argv;
		if ( output_folder.compare(output_folder.size()-1, 1, "/")!=0 )
			output_folder.append("/");
#ifdef _WIN32

		if ( _mkdir(output_folder.c_str()) != 0 ) {
			if (errno == ENOENT) {
				if (verbose) printf("ERROR: unable to create results folder %s", output_folder.c_str());
				if (verbose) perror("");
				output_folder = "./";
			}
		}
#else
		mode_t process_mask = umask(0);
		if (mkdir(output_folder.c_str(), S_IRWXU|S_IRGRP|S_IXGRP) != 0 ) {
		// if ( mkdir(output_folder.c_str(), != 0, S_IRWXU | S_IRWXG | S_IRWXO ) ) {
			// if (errno != 0) {
				if (verbose) printf("ERROR: unable to create results folder %s", output_folder.c_str());
				if (verbose) perror("");
				output_folder = "./";
			// }
		}
		umask(process_mask);
#endif 

	}	

	return output_folder;

}


bool work_completed(unsigned int PARALLEL_THREADS, unsigned int PARALLEL_BLOCKS, unsigned int GPU ) {
	// cudaMemcpy(verifica, dev_perblock_ended, sizeof(char)*PARALLEL_BLOCKS, cudaMemcpyDeviceToHost);
	cudaMemcpy(verifica, dev_perblock_ended, sizeof(int)*PARALLEL_BLOCKS, cudaMemcpyDeviceToHost);
	CudaCheckError();
	for(unsigned int i=0; i<PARALLEL_THREADS; i++)
		printf("%d/%d\n", verifica[i], -(PARALLEL_THREADS*PARALLEL_BLOCKS));
	return verifica[0]==-(PARALLEL_THREADS*PARALLEL_BLOCKS);
}

/* ON CPU */
bool work_completed_old(unsigned int PARALLEL_THREADS, unsigned int PARALLEL_BLOCKS, unsigned int GPU ) {

	// char* verifica = (char*) malloc( PARALLEL_BLOCKS*sizeof(char));
	
	int finiti = 0;
	static int status = INT_MAX;

	// cudaMemcpy(verifica, dev_perthread_SSA, PARALLEL_BLOCKS*sizeof(char), cudaMemcpyDeviceToHost);
	cudaMemcpy(verifica, dev_perblock_ended, sizeof(char)*PARALLEL_BLOCKS, cudaMemcpyDeviceToHost);
	CudaCheckError();
			
	
	for (unsigned int i=0; i<PARALLEL_BLOCKS; i++)  {
		if (verifica[i]==-PARALLEL_THREADS) {
			finiti++;			
		}		
	}
	

	if (status!=finiti) {
		status = finiti;
		printf("Status: %d blocks out of %d (%s) on GPU %d\n", status, PARALLEL_BLOCKS, get_date_dump().c_str(), GPU);
		ReadSSA( PARALLEL_THREADS, PARALLEL_BLOCKS);
		for (unsigned int i=0; i<PARALLEL_BLOCKS; i++) 
			printf("Reduced %d\n", verifica[i]);
		printf("\n");
		
	}

	// free( verifica );
	// verifica = NULL;
	
	// return true;

	if (finiti == PARALLEL_BLOCKS)	return true;
	return false;
	// return conclusioni==(int)PARALLEL_BLOCKS*-1 ;
}

void start_profiling(cudaEvent_t* start, cudaEvent_t* stop) {

	/// TIMER 1	
	cudaEventCreate(start);  
	cudaEventCreate(stop);
	cudaEventRecord(*start, 0);

}

float stop_profiling(cudaEvent_t* start, cudaEvent_t* stop) {

	cudaEventRecord( *stop, 0 );
	cudaEventSynchronize( *stop );
	float tempo;
	cudaEventElapsedTime( &tempo, *start, *stop );
	tempo /= 1000;
	printf(" * RUNNING TIME: %f seconds.\n", tempo);
	/*
	std::ofstream runningtime("simulation_time");
	runningtime << tempo ;
	runningtime.close();
	*/
	return tempo;

}

/*
void reduce( dim3 blocksPerGrid, dim3 threadsPerBlock, size_t sharedmem,  char* dev_perthread_SSA, int* dev_perblock_ended, int n) {

	sharedmem = (threadsPerBlock.x <= 32) ? 2 * threadsPerBlock.x * sizeof(int) : threadsPerBlock.x * sizeof(int);

	switch( threadsPerBlock.x ) {
			case 256:
				reduce7< 256 >
					<<<blocksPerGrid, threadsPerBlock, sharedmem >>>
					(dev_perthread_SSA, dev_perblock_ended, n );
				break;
			case 128:
				reduce7< 128 >
					<<<blocksPerGrid, threadsPerBlock, sharedmem >>>
					(dev_perthread_SSA, dev_perblock_ended, n );
				break;
			case 64:
				reduce7< 64 >
					<<<blocksPerGrid, threadsPerBlock, sharedmem >>>
					(dev_perthread_SSA, dev_perblock_ended, n );
				break;
			case 32:
				reduce7< 32 >
					<<<blocksPerGrid, threadsPerBlock, sharedmem >>>
					(dev_perthread_SSA, dev_perblock_ended, n );
				break;
			case 16:
				reduce7< 16 >
					<<<blocksPerGrid, threadsPerBlock, sharedmem >>>
					(dev_perthread_SSA, dev_perblock_ended, n );
				break;
			case 8:
				reduce7< 8 >
					<<<blocksPerGrid, threadsPerBlock, sharedmem >>>
					(dev_perthread_SSA, dev_perblock_ended, n );
				break;
			case 4:
				reduce7< 4 >
					<<<blocksPerGrid, threadsPerBlock, sharedmem >>>
					(dev_perthread_SSA, dev_perblock_ended, n );
				break;
			case 2:
				reduce7< 2 >
					<<<blocksPerGrid, threadsPerBlock, sharedmem >>>
					(dev_perthread_SSA, dev_perblock_ended, n );
				break;
			case 1:
				reduce7< 1 >
					<<<blocksPerGrid, threadsPerBlock, sharedmem >>>
					(dev_perthread_SSA, dev_perblock_ended, n );
				break;
		}
		
		CudaCheckError();
		
		cudaThreadSynchronize();
}
*/


void ReadSSA( unsigned int tpb, unsigned int blocks) {

	cudaThreadSynchronize();
	cudaMemcpy(host_SSA, dev_perthread_SSA,  tpb*blocks*sizeof(char), cudaMemcpyDeviceToHost);
	for (unsigned int b=0; b<blocks; b++) {
		for (unsigned int t=0; t<tpb; t++) {
			printf("%d-%d:%d\t", b,t,host_SSA[tpb*b+t]);
		}
		printf("\n");
	}


}





//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////// FITNESS standard //////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	
	
__device__ inline  int fetch_target(unsigned int* source, unsigned int species, unsigned int experiment, 
									unsigned int repetition, unsigned int sample) {
	return source[ 
		sample*(DEV_CONST_SAMPLESPECIES*DEV_CONST_EXPERIMENTS*DEV_CONST_REPETITIONS) + 
			experiment*(DEV_CONST_SAMPLESPECIES*DEV_CONST_REPETITIONS) + 
				repetition*(DEV_CONST_SAMPLESPECIES) + species ];
}

/*
__device__ inline  int fetch_simulation(unsigned int* source, unsigned int species, unsigned int sample) {
	unsigned int gid = threadIdx.x + blockDim.x*blockIdx.x;	
 	return source[ blockDim.x*gridDim.x*DEV_CONST_SAMPLESLUN*sample + blockDim.x*gridDim.x*species+ gid ];	
}
*/

__global__ void calculateFitness( unsigned int* samples, unsigned int * target, double* fitness, char* swarm, float* normalizers ) {

	unsigned int tid = threadIdx.x + blockDim.x*blockIdx.x;

	unsigned int D = DEV_CONST_EXPERIMENTS;		// dosi
	unsigned int E = DEV_CONST_REPETITIONS;		// ripetizioni ad ogni dose
	unsigned int experiment = swarm[tid]; 		// "dose" <- lanciamo un numero di blocchi pari agli swarm

	float subfitness = 0;
	
	/*
	 if (tid==0) {
				printf("%d, %d, %d.\n", DEV_CONST_SAMPLESPECIES, DEV_CONST_EXPERIMENTS, DEV_CONST_REPETITIONS);
                for (unsigned int c=0; c<31; c++) {
                        for (unsigned int s=0; s<4; s++) {
                                printf("%d (%d), ", target[c*4+s], fetch_target(target, s, 0, 0, c));
                        }
                }
                printf("\n");
        }
     return;
	*/

	// for each sample...
	for (unsigned int campione=0; campione<DEV_CONST_SAMPLES; campione++) {

		// ...for each (sampled) chemical species...
		for (unsigned int s=0; s<DEV_CONST_SAMPLESPECIES; s++) {

			// float norm_value = normalizers[experiment * (DEV_CONST_SAMPLESPECIES+1) +s];

			double accum = 0;

			// ...calculate the average of the parallel simulations...
			for (unsigned int par=0; par<DEV_CONST_PAR_REPETITIONS; par++ ) {

				accum += 
					samples [

						blockDim.x * (gridDim.x*DEV_CONST_PAR_REPETITIONS) * DEV_CONST_SAMPLESPECIES * campione + 
						blockDim.x * (gridDim.x*DEV_CONST_PAR_REPETITIONS) * s + 
						blockDim.x * (blockIdx.x * DEV_CONST_PAR_REPETITIONS + par) + 
						threadIdx.x

					];

			} // end for par

			accum /= DEV_CONST_PAR_REPETITIONS;

			// calculate the distance from target time series
			
			for (unsigned int repetition=0; repetition<E; repetition++) {
				int tgt = fetch_target(target, s, experiment, repetition, campione);
				// if(tid==0) printf("s:%d, e:%d, r:%d, c:%d, %d.\n", s, experiment, repetition, campione, tgt);
				double dist = (double)abs((double)tgt-accum);
				if (tgt!=0) dist /= tgt;
				// subfitness += (dist * norm_value);
				subfitness += (dist);
				// if (tid==1) 		printf("Tid%d\tBid%d\tExp%d\tRep%d\tSam%d\tSpe%d\tTar%u-%f\tDis%f\tSub%f\tNor%f\n", 
				//threadIdx.x, blockIdx.x, experiment, repetition, campione, s, tgt, accum, dist, subfitness, norm_value);				
			}
			
			/*
			for (unsigned int repetition=0; repetition<E; repetition++) {
				int tgt = fetch_target(target, s, experiment, repetition, campione);
				double dist =
					tgt == accum?
						0 : log((double)abs(tgt-accum));
				// if (tgt!=0) dist /= tgt;
				subfitness += dist;
				// if (tid==1) 					printf("%d\t%d\t%d\t%d\t%d\t%d\t%u-%f\t%f\t%f\n", threadIdx.x, blockIdx.x, experiment, repetition, campione, s, tgt, accum, dist, subfitness);				
			}
			*/
		}

	}
	
	// divisione per il numero di specie e per il numero di campioni
	fitness[tid] = (1.0/DEV_CONST_TIMESLUN)*(1.0/DEV_CONST_SAMPLESPECIES)*subfitness; 

	// DEBUG
	// fitness[tid] = tid;
	
};


__global__ void test_dump(unsigned int* dev_per_thread_storage, unsigned threads, unsigned blocks, unsigned species, unsigned samples) {
	printf("DUMPA LUMPA\n");
	
	for (unsigned bl=30; bl<blocks; bl++) {
		printf("BLK%d\n", bl);
		for (unsigned tr=0; tr<threads; tr++) {
			printf("Tr%d\n) ", tr);
			for (unsigned sa=0; sa<samples; sa++) {
				printf("Spl%d ", sa);
				for (unsigned sp=0; sp<species; sp++) {
					printf("%d\t",dev_per_thread_storage[ blocks*threads*species*sa + blocks*threads*sp + threads*bl+tr ]);
				}
			}
			printf("\n");
		}
	}

};


void WriteDynamics2( std::string d, unsigned int parallel_threads, unsigned int pblocks, unsigned int samples, unsigned int species, unsigned int samplespecies ) {

	cudaThreadSynchronize();

	unsigned int larg =  parallel_threads * pblocks;

	unsigned int bytesize =  sizeof(unsigned int ) * larg * samples * samplespecies ;

	
	unsigned int * host_dynamics = (unsigned int *) malloc (bytesize);
	memset(host_dynamics, 0, bytesize);
	cudaMemcpy( host_dynamics, dev_perthread_storage, bytesize, cudaMemcpyDeviceToHost);
	CudaCheckError();

	unsigned int DEV_CONST_SPECIES = species;

	for (unsigned int gid=0; gid<parallel_threads*pblocks; gid++) {		

		std::ofstream myfile;
		std::stringstream ss; ss << gid;
		std::string nome_file;
		nome_file = d;
		nome_file.append("_");
		nome_file.append(ss.str());
		myfile.open (nome_file.c_str() );
		
		for (unsigned int i=0; i<samples; i++ ) {			
			myfile << host_sampletime[i]; 
			for (unsigned int s=0; s<samplespecies; s++) {
				myfile << "\t" << host_dynamics[ larg*samplespecies*i + larg*s + gid ];
			}
			myfile << "\n";
		}
		myfile.close();
		// myfile << "\n";
	}
	

}
