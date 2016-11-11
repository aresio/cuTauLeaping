#cuTauLeaping release 1.0.0

## ABOUT

cuTauLeaping is a stochastic simulator of biological systems that exploits the remarkable memory bandwidth and computational capability of GPUs. 
cuTauLeaping allows to efficiently execute in parallel large numbers of stochastic simulations, which are usually required to investigate the emergent dynamics of a given biological system under different conditions.
cuTauLeaping is based on Cao's improved version of tau-leaping (https://dx.doi.org/10.1063%2F1.2159468). 


## DEPENDENCIES

Just the Nvidia CUDA library (version >7.0).


##  COMPILATION

A cuTauLeaping binary can be compiled on any supported architecture (e.g., GNU/Linux, Microsoft Windows, Apple OS/X) using the following compilation command:

```bash
nvcc kernel.cu 2phase_n-tau-leaping.cu -gencode=arch=compute_20,code=compute_20 -O3 -o cuTauLeaping --use_fast_math
```

The command above would create a binary executable file runnable on GPUs with _at least_ a compute capability equal to 2.0. Please note that a specific compute capability, supporting additional functionality, can be targeted by using the ```gencode``` argument. For instance, to target the compute capability 3.5, the following argument can be passed to ```nvcc```:

```bash
-gencode=arch=compute_35,code=compute_35
```

## LAUNCHING CUTAULEAPING

cuTauLeaping is designed to be launched from the command line. The arguments are: 

`cupSODA input_folder threads blocks gpu offset  output_folder prefix fitness force_ssa`

where

* `input_folder` is the path to the directory containing the input model;
* `threads` is the number of CUDA threads per blocks;
* `blocks` is the number of CUDA blocks used to distribute the requested parallel threads;
* `output_folder` is the path to the directory that will store the output dynamics of the simulations;
* `prefix` is the file name of the output files. A number, corresponding the thread, will be automatically appended to the filename by cupSODA;
* `force_ssa` forces the algorithm to rely only on exact SSA steps.

Further information about the `gpu`, `fitness` and `offset` arguments, along with the specifications of the input files, can be found at the following address:

https://docs.google.com/document/d/1gPq-mYk-IP-bVmiMZewGPmTJ6nMCH8al1nNr7OaBsv4/edit?usp=sharing

## LICENSE

BSD License


## CONTACT 

nobile@disco.unimib.it
