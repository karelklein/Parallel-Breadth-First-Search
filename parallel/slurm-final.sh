#!/bin/sh
#SBATCH  -J bfs                          # Job name
#SBATCH  -p development                  # Queue (development or normal)
#SBATCH  -N 1                            # Number of nodes
#SBATCH --tasks-per-node 4               # Number of tasks per node
#SBATCH  -t 00:10:00                     # Time limit hrs:min:sec
#SBATCH  -A TG-TRA170035                 # Allocation
#SBATCH  -o bfs=%j.out                   # Standard output and error log

module use /home1/01236/tisaac/opt/modulefiles
module load petsc/cse6230-double

make

make test_bfs

git rev-parse HEAD

git diff-files

pwd; hostname; date

ibrun tacc_affinity ./test_bfs -tests 4 -num_time 10000

date

