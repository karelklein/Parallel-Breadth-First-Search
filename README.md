# Parallel Breadth First Search

An implementation of parallel BFS using the OpenMP API for shared memory multiprocessing.

## Computing Platform
- TACC's Stampede2 supercomputer

## Contents
- /parallel
	- Code: bfs.c
	- Performance profiling: performance_profiling.pdf
	- Scaling study: final_analysis.pdf

- /serial
	- Code: bfs.c
	- Algorithm analysis: report.pdf

## Running Code in Stampede2
1. `cd parallel/`
2. `./slurm_final.sh`