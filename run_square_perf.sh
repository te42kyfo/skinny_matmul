#!/bin/sh

xrange=11
yrange=11

for (( x=1 ; x<$xrange; x+=3 ))
do
    for (( y=1 ; y<$yrange; y+=2 ))
    do
        make perf  M=$x N=$y GENVER=$1 PREFIX=./build 2>&1  &
        while test $(jobs -p | wc -w) -ge 100; do sleep 1; done
    done
done

wait
echo "all built"

for (( x=1 ; x<$xrange; x+=3 ))
do
    for (( y=1 ; y<$yrange; y+=2 ))
    do
        ./build/perf$x-$y-$1
    done
done
