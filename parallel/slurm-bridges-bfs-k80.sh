#!/bin/sh
#SBATCH  -J bfs-k80                      # Job name
#SBATCH  -p GPU-shared                   # Queue (RM, RM-shared, GPU, GPU-shared)
#SBATCH  -N 1                            # Number of nodes
#SBATCH --gres=gpu:k80:4                 # GPU type and amount
#SBATCH  -t 00:05:00                     # Time limit hrs:min:sec
#SBATCH  -o bfs-k80-%j.out               # Standard output and error log

module use /home/tisaac/opt/modulesfiles
module load petsc/cse6230-double

if [ ! -f Makefile.cuda ]; then
  echo "MMMA_CUDA = 1" > Makefile.cuda
fi

make test_bfs

git rev-parse HEAD

git diff-files

pwd; hostname; date

./test_bfs

date
