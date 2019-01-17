#!/bin/bash

make test_bfs

git rev-parse HEAD

git diff-files

pwd; hostname; date

mpiexec -n 4 ./test_bfs -tests 1,2,3,4 -num_time 30

date
