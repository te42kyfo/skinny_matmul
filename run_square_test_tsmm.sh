#!/bin/sh


if [ -z $1 ] || [ -z $2 ] || [ -z $3 ]; then
    echo "Usage: $0 <mode> <xrange> <yrange>"
    exit
fi

xrange=$2
yrange=$3

for (( x=1 ; x<=$xrange; x+=1 ))
do
    for (( y=1 ; y<=$yrange; y+=1 ))
    do
        make test_tsmm  M=$x N=$y MODE=$1 PREFIX=./build 2>&1  &
        while test $(jobs -p | wc -w) -ge 100; do sleep 1; done
    done
done

wait
echo "all built"

for (( x=1 ; x<=$xrange; x+=1 ))
do
    for (( y=1 ; y<=$yrange; y+=1 ))
    do
        ./build/test_tsmm$x-$y-$1
    done
done