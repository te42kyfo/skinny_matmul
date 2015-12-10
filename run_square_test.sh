#!/bin/sh

xrange=64
yrange=64

for (( x=1 ; x<$xrange; x+=1 ))
do
    for (( y=1 ; y<$yrange; y+=1 ))
    do
        make test  M=$x N=$y GENVER=$1 PREFIX=./build 2>&1  &
        while test $(jobs -p | wc -w) -ge 100; do sleep 1; done
    done
done

wait
echo "all built"

for (( x=1 ; x<$xrange; x+=1 ))
do
    for (( y=1 ; y<$yrange; y+=1 ))
    do
        ./build/test$x-$y-$1
    done
done
