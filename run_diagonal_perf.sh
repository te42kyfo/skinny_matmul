#!/bin/sh

if [ -z $1 ] || [ -z $2 ] || [ -z $3 ]; then
    echo "Usage: $0 [tsmttsm/tsmm] <types> <x1> <x2>"
    exit
fi


multype=$1
dtype=$2
xrange1=$3
xrange2=$4
if  [ -z $4 ]; then
    xrange1=1
    xrange2=$3
fi

for (( x=$xrange1 ; x<=$xrange2; x+=1 ))
do
    make perf M=$x N=$x MULTYPE=$multype TYPES=$dtype PREFIX=./build 1>&2  &
    while test $(jobs -p | wc -w) -ge 40; do sleep 1; done
done

wait

for (( x=$xrange1 ; x<=$xrange2; x+=1 ))
do
    ./build/perf-$multype-$x-$x-$dtype
done
