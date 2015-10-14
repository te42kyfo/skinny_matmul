#!/bin/bash -l


module load gcc/4.8.1
module load cuda/6.5

mkdir build

for m in {1..5}
do
    for n in {1..5}
    do
        make perf N=$n M=$m GENVER=$3 PREFIX=./build &
        while test $(jobs -p | wc -w) -ge 200; do sleep 1; done
    done
done

wait
echo "all done"

rm square_$2_$3.txt

for m in {1..5}
do
    for n in {1..5}
    do
        ./build/perf$m-$n-$3 | tee --append ./square_$2_$3.txt
    done
done