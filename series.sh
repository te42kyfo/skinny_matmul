#!/bin/sh

git clone . $1/$2

cd $1/$2

git checkout $2

mkdir build

for m in {1..100}
do
    for n in {1..100}
    do
        make  perf N=$n M=$m PREFIX=./build &
        while test $(jobs -p | wc -w) -ge 200; do sleep 1; done
    done
done

wait
echo "all done"

rm square.txt

for m in {1..100}
do
    for n in {1..100}
    do
        ./build/perf$m-$n | tee --append ./square.txt
    done
done
