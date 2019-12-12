#!/bin/bash
module add mpi/openmpi4-x86_64
mpiexec  python3 script.py -d ./data/data.csv -l ./data/label.csv -s 1000

