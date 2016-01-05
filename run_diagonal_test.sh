#!/bin/sh

range=32

if [ -z $1 ]; then
    echo "Specify Kernel Version"
    exit
fi

for (( d=1 ; d<$range; d++ ))
do
    make test M=$d N=$d GENVER=$1 PREFIX=./build 2>&1  &
    while test $(jobs -p | wc -w) -ge 100; do sleep 1; done
done


wait
echo "all built"

for (( d=1 ; d<$range; d++ ))
do
    ./build/test$d-$d-$1
done

