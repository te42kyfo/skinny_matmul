#!/bin/sh



if [ -z $1 ] || [ -z $2 ]; then
    echo "Usage: $0 <Kernel Version> <drange>"
    exit
fi

range=$2

make perf  M=0 N=0 GENVER=$1 PREFIX=./build 1>&2  &

for (( d=1 ; d<=$range; d++ ))
do
    make perf M=$d N=$d GENVER=$1 PREFIX=./build 1>&2 &
    while test $(jobs -p | wc -w) -ge 100; do sleep 1; done
done


wait

./build/perf0-0-$1

for (( d=1 ; d<=$range; d++ ))
do
    ./build/perf$d-$d-$1
done

