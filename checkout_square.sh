#!/bin/sh

if [ -z $1 ] || [ -z $2 ] || [ -z $3 ] || [ -z $4 ] || [ -z $5 ] || [ -z $6 ]; then
    echo "Usage: $0 <prefix> <commit hash> <kernel version> <mode> <xrange> <yrange>"
    exit
fi

git clone . $1/$2

cd $1/$2

git checkout $2

mkdir build

rm square_$2_$3_$4.txt

./run_square_perf.sh $3 $4 $5 $6 | tee --append square_$2_$3_$4.txt

