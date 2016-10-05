#!/bin/sh

if [ -z $1 ] || [ -z $2 ] || [ -z $3 ] || [ -z $3 ]; then
    echo "Usage: $0 [tsmm/tsmttsm] <Kernel Version> <Mode> <drange>"
    exit
fi

cd $1

multype=$1
ver=$2
dtype=$3
range=$4

make perf  M=0 N=0 GENVER=$ver MODE=$dtype PREFIX=./build 1>&2  &

for (( d=1 ; d<=$range; d++ ))
do
    make perf M=$d N=$d GENVER=$ver MODE=$dtype PREFIX=./build 1>&2 &
    while test $(jobs -p | wc -w) -ge 100; do sleep 1; done
done


wait

./build/perf0-0-$dtype

for (( d=1 ; d<=$range; d++ ))
do
    ./build/perf$d-$d-$dtype
done

