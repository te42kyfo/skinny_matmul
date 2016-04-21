#!/bin/sh

if [ -z $1 ] || [ -z $2 ] || [ -z $3 ] || [ -z $4 ]; then
    echo "Usage: $0 <Kernel Version> <mode> <xrange> <yrange>"
    exit
fi

xrange=$3
yrange=$4

make perf  M=0 N=0 GENVER=$1 MODE=$2 PREFIX=./build 1>&2  &

for (( x=1 ; x<=$xrange; x+=1 ))
do
    for (( y=1 ; y<=$yrange; y+=1 ))
    do
        make perf  M=$x N=$y GENVER=$1 MODE=$2 PREFIX=./build 1>&2  &
        while test $(jobs -p | wc -w) -ge 100; do sleep 1; done
    done
done

wait

./build/perf0-0-$1-$2

for (( x=1 ; x<=$xrange; x+=1 ))
do
    for (( y=1 ; y<=$yrange; y+=1 ))
    do
        ./build/perf$x-$y-$1-$2
    done
done
