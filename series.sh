#!/bin/sh

git clone . $1/$2

mkdir $1/$2/build

for m in {1..10}
do
    for n in {1..10}
    do
        make -f $1/$2/Makefile perf N=$n M=$m PREFIX=$1/$2/build &
        while test $(jobs -p | wc -w) -ge 200; do sleep 1; done
    done
done

wait
echo "all done"

rm square.txt

for m in {1..10}
do
    for n in {1..10}
    do
        $1/$2/build/perf$m-$n | tee --append square.txt
    done
done
