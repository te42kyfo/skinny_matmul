#!/bin/sh

if [ -z $1 ] || [ -z $2 ] || [ -z $3 ]; then
    echo "Usage: $0 <prefix> <commit hash> <kernel version>"
    exit
fi

git clone . $1/$2

cd $1/$2

git checkout $2

mkdir build

rm square_$2_$3.txt

./run_square_perf.sh $3 32 32 | tee --append square_$2_$3.txt

