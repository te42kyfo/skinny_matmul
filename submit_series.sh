#!/bin/sh

git clone . $1/$2

cd $1/$2

git checkout $2

qsub square_pbs.sh -l nodes=1:ppn=40:k40m -l walltime=8:00:00 -N square_$2_$3 -F "$1 $2 $3" 
