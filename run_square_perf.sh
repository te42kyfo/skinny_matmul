#!/bin/sh

if [ -z $1 ] || [ -z $2 ] || [ -z $3 ]; then
    echo "Usage: $0 <Kernel Version> <xrange> <yrange>"
    exit
fi

xrange=$2
yrange=$3

make perf  M=0 N=0 GENVER=$1 PREFIX=./build 1>&2  &

for (( x=1 ; x<=$xrange; x+=1 ))
do
    for (( y=1 ; y<=$yrange; y+=1 ))
    do
        make perf  M=$x N=$y GENVER=$1 PREFIX=./build 1>&2  &
        while test $(jobs -p | wc -w) -ge 100; do sleep 1; done
    done
done

wait

./build/perf0-0-$1

for (( x=1 ; x<=$xrange; x+=1 ))
do
    for (( y=1 ; y<=$yrange; y+=1 ))
    do
        ./build/perf$x-$y-$1
    done
done
