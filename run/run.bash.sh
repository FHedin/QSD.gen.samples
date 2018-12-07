#!/bin/bash

ORIG_DIR=$PWD
#SCRATCH_DIR=$HOME/libre/

#TMP_DIR=$(mktemp --tmpdir=$SCRATCH_DIR -u)
TMP_DIR=$(mktemp --tmpdir=$ORIG_DIR -u)

mkdir -p $TMP_DIR && echo "Running in tmp dir $TMP_DIR"

cd $TMP_DIR

ln -s $ORIG_DIR/mol
ln -s $ORIG_DIR/mol/ala2vac/input.lua
ln -s $ORIG_DIR/QSD.gen.samples

EXEC=./QSD.gen.samples

# set this variable to the number of cpu cores you want to allow for each replica
export OPENMM_CPU_THREADS=1
# required for the executable to locate OpenMM plugins
export OPENMM_PLUGIN_DIR=$ORIG_DIR/../build/openmm-7.3.0/lib/plugins

# running with 8 replicas (much much more expected for a production run !!)
#  the first rep prints everything to the terminal
#  the 7 following to a text file : out.txt and err.txt are template names, rep 1 will write to out.1.txt, rep 2 to out.2.txt ...
mpirun -np 1  $EXEC -i input.lua -log dbg : \
              -np 7 $EXEC -i input.lua -log dbg -o out.txt -e err.txt

cd $ORIG_DIR

#ln -s $TMP_DIR

