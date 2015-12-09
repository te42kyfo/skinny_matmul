#!/bin/sh

xrange=5
yrange=5

for (( x=1 ; x<$xrange; x++ ))
do
    for (( y=1 ; y<$yrange; y++ ))
    do
        make test  M=$x N=$y GENVER=$1 PREFIX=./build 2>&1  &
        while test $(jobs -p | wc -w) -ge 300; do sleep 1; done
    done
done

wait
echo "all built"

for (( x=1 ; x<$xrange; x++ ))
do
    for (( y=1 ; y<$yrange; y++ ))
    do
        ./build/test$x-$y-$1
    done
done
