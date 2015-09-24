#!/bin/sh

git clone . $1/$2

cd $1/$2

git checkout $2

mkdir build


for d in {1..100}
do
    make  perf N=$d M=$d GENVER=$3 PREFIX=./build &
    while test $(jobs -p | wc -w) -ge 200; do sleep 1; done
done


wait
echo "all done"

rm diag_$2_$3.txt

for d in {1..100}
do
    ./build/perf$d-$d-$3 | tee --append ./diag_$2_$3.txt
done

