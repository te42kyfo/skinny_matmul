#!/bin/bash

rm sbatch_output.txt

sbatch jobscript.sh


while [ ! -f sbatch_output.txt ]
do
      squeue | grep dernst
      sleep 3
done

tail -f sbatch_output.txt
