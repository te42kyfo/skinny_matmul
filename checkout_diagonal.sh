#!/bin/sh

if [ -z $1 ] || [ -z $2 ] || [ -z $3 ]; then
    echo "Usage: $0 <prefix> <commit hash> <kernel version>"
    exit
fi

git clone . $1/$2

cd $1/$2

git checkout $2

mkdir build

rm diag_$2_$3.txt

./run_diagonal_perf.sh $3 100 | tee --append diag_$2_$3.txt


