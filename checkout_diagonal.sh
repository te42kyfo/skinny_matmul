#!/bin/sh

if [ -z $1 ] || [ -z $2 ] || [ -z $3 ] || [ -z $4 ]  || [ -z $5 ]; then
    echo "Usage: $0 <prefix> <commit hash> <kernel version> <mode> <range>"
    exit
fi

git clone . $1/$2

cd $1/$2

git checkout $2

mkdir build

rm diag_$2_$3_$4.txt

./run_diagonal_perf.sh $3 $4 $5 | tee --append diag_$2_$3_$4.txt


