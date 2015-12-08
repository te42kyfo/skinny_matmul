#!/bin/sh

range=32


for (( d=1 ; d<$range; d++ ))
do
    make test N=$d M=$d GENVER=$3 PREFIX=./build 2>&1 > /dev/null &
    while test $(jobs -p | wc -w) -ge 100; do sleep 1; done
done

while test $(jobs -p | wc -w) -ge 2; do
    sleep 1
    completed=d-$(jobs -p | wc -w)
    echo -en '\r|'
    for (( k=1 ; k<$completed; k++ ))
    do
        echo -n "."
    done
    for (( k=$completed+1; k<$d; k++ ))
    do
        echo -n " "
    done
    echo -n "|"
done

echo

wait
echo "all built"

for (( d=1 ; d<$range; d++ ))
do
    ./build/test$d-$d-$3
done

