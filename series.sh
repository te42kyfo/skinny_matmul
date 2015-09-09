#!/bin/sh

git clone . $1/$2

mkdir $1/$2/build

for m in {10..30}
do
    for n in {2..2}
    do
        make -f $1/$2/Makefile perf N=$n M=$m PREFIX=$1/$2/build &
        while test $(jobs -p | wc -w) -ge 200; do sleep 1; done
    done
done

wait
echo "all done"

rm $1/$2/square.txt

for m in {10..30}
do
    for n in {2..2}
    do
        $1/$2/build/perf$m-$n | tee --append $1/$2/square.txt
    done
done
