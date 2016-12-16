#!/bin/sh

if [ -z $1 ] || [ -z $2 ] || [ -z $3 ] || [ -z $4 ]; then
    echo "Usage: $0 [tsmttsm/tsmm] <types> <xrange> <yrange>"
    exit
fi

multype=$1
dtype=$2
xrange=$3
yrange=$4

for (( x=1 ; x<=$xrange; x+=1 ))
do
    for (( y=1 ; y<=$yrange; y+=1 ))
    do
        make test  M=$x N=$y MULTYPE=$multype TYPES=$dtype PREFIX=./build 1>&2  &
        while test $(jobs -p | wc -w) -ge 60; do sleep 1; done
    done
done

wait

for (( x=1 ; x<=$xrange; x+=1 ))
do
    for (( y=1 ; y<=$yrange; y+=1 ))
    do
        ./build/test-$multype-$x-$y-$dtype
    done
done
