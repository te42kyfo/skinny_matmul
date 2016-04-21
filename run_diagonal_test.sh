#!/bin/sh


if [ -z $1 ] || [ -z $2 ] || [ -z $3 ]; then
    echo "Usage: $0 <Kernel Version> <Mode> <drange>"
    exit
fi

range=$3

for (( d=1 ; d<$range; d++ ))
do
    make test M=$d N=$d GENVER=$1 MODE=$2 PREFIX=./build 2>&1  &
    while test $(jobs -p | wc -w) -ge 100; do sleep 1; done
done


wait
echo "all built"

for (( d=1 ; d<$range; d++ ))
do
    ./build/test$d-$d-$1-$2
done

