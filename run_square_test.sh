#!/bin/sh

xrange=16
yrange=16

if [ -z $1 ]; then
    echo "Specify Kernel Version"
    exit
fi

for (( x=1 ; x<=$xrange; x+=1 ))
do
    for (( y=1 ; y<=$yrange; y+=1 ))
    do
        make test  M=$x N=$y GENVER=$1 PREFIX=./build 2>&1  &
        while test $(jobs -p | wc -w) -ge 100; do sleep 1; done
    done
done

wait
echo "all built"

for (( x=1 ; x<=$xrange; x+=1 ))
do
    for (( y=1 ; y<=$yrange; y+=1 ))
    do
        ./build/test$x-$y-$1
    done
done
