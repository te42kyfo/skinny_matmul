#!/bin/sh

if [ -z $1 ] || [ -z $2 ] || [ -z $3 ] || [ -z $4 ] || [ -z $4 ] || [ -z $5 ]; then
    echo "Usage: $0 [tsmttsm/tsmm] <Kernel Version> <mode> <xrange> <yrange>"
    exit
fi

cd $1

multype=$1
ver=$2
dtype=$3
xrange=$4
yrange=$5

make perf  M=0 N=0 GENVER=$ver MODE=$dtype PREFIX=./build 1>&2  &

for (( x=1 ; x<=$xrange; x+=1 ))
do
    for (( y=1 ; y<=$yrange; y+=1 ))
    do
        make perf  M=$x N=$y GENVER=$ver MODE=$dtype PREFIX=./build 1>&2  &
        while test $(jobs -p | wc -w) -ge 100; do sleep 1; done
    done
done

wait

./build/perf0-0-$dtype

for (( x=1 ; x<=$xrange; x+=1 ))
do
    for (( y=1 ; y<=$yrange; y+=1 ))
    do
        ./build/perf$x-$y-$dtype
    done
done
